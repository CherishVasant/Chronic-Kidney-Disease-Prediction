from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import os
from scipy.stats import zscore
import xgboost
import warnings
warnings.filterwarnings('ignore')

# Setup logging with format and console output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler()  # This will send logs to terminal
    ]
)

# Define logger object
logger = logging.getLogger(__name__)

app = Flask(__name__)

logger.info("Flask app started!")

# Global variable to store the model
model = None

# Create a separate dataframe for engineered features
engineered_features = {}
patient_data = {}

def load_model():
    """Load the trained model from pickle file"""
    global model
    try:
        with open('v1_CatBoost.pkl', 'rb') as file:
            model = pickle.load(file)
            logger.info("Model loaded successfully")

        return True
    except FileNotFoundError:
        logger.error("Model file 'v1_CatBoost.pkl' not found")
        return False
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False



def preprocess_input(form_data):
    """
    Preprocess form data using pre-fitted OneHotEncoder and StandardScaler
    Returns a DataFrame with engineered features and properly scaled/encoded data.
    """
    
    # Load the original training data for z-score calculations
    try:
        with open('data.pkl', 'rb') as f:
            orig_data = pickle.load(f)
    except FileNotFoundError:
        logger.error("data.pkl file not found")
        raise
    except Exception as e:
        logger.error(f"Error loading training data: {str(e)}")
        raise

    # Load pre-fitted scaler
    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        logger.error("scaler.pkl file not found")
        raise
    except Exception as e:
        logger.error(f"Error loading scaler: {str(e)}")
        raise

    # Column lists used during training
    num_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
    cat_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

    try:
        # Process numerical columns with validation
        for col in num_cols:
            val = form_data.get(col, 0)
            try:
                patient_data[col] = float(val)
            except (ValueError, TypeError):
                logger.warning(f"Invalid numerical value for {col}: {val}, using 0")
                patient_data[col] = 0.0

        # Process categorical columns with validation
        for col in cat_cols:
            val = form_data.get(col, 'unknown')
            patient_data[col] = str(val) if val is not None else 'unknown'

        # Create DataFrame from processed input data (original values)
        df = pd.DataFrame([patient_data])
        logger.info(f"Patient data: {patient_data}")

        # Feature Engineering section
        
        # 1. eGFR calculation
        sc = df['sc'].iloc[0]
        age = df['age'].iloc[0]
        
        # Validate inputs for eGFR calculation
        sc = max(sc, 0.01)  # Avoid division by zero
        age = max(age, 1)   # Avoid invalid age
        
        egfr = 186 * (sc ** (-1.154)) * (age ** (-0.203))
        engineered_features['eGFR'] = egfr
        
        # 2. Comorbidity score
        htn = 1 if df['htn'].iloc[0] == 'yes' else 0
        dm = 1 if df['dm'].iloc[0] == 'yes' else 0
        cad = 1 if df['cad'].iloc[0] == 'yes' else 0
        
        comorb = htn + dm + cad
        engineered_features['comorb_score'] = comorb
        
        # 3. Anemia severity using z-scores from original dataset
        hemo = df['hemo'].iloc[0]
        pcv = df['pcv'].iloc[0]
        rc = df['rc'].iloc[0]

        # Calculate z-scores using the original training dataset statistics
        hemo_mean, hemo_std = orig_data['hemo'].mean(), orig_data['hemo'].std()
        pcv_mean, pcv_std = orig_data['pcv'].mean(), orig_data['pcv'].std()
        rc_mean, rc_std = orig_data['rc'].mean(), orig_data['rc'].std()

        # Validate standard deviations to avoid division by zero
        if hemo_std == 0 or pcv_std == 0 or rc_std == 0:
            logger.warning("Zero standard deviation detected in training data")
            hemo_std = max(hemo_std, 1e-6)
            pcv_std = max(pcv_std, 1e-6)
            rc_std = max(rc_std, 1e-6)

        hemo_z = (hemo - hemo_mean) / hemo_std
        pcv_z = (pcv - pcv_mean) / pcv_std
        rc_z = (rc - rc_mean) / rc_std

        anemia_sev = -(hemo_z + pcv_z + rc_z)
        engineered_features['anemia_severity'] = anemia_sev
        
        # 4. Kidney function score using z-scores from original dataset
        bu = df['bu'].iloc[0]
        sc_orig = df['sc'].iloc[0]
        sod = df['sod'].iloc[0]

        # Calculate z-scores using the original training dataset statistics
        bu_mean, bu_std = orig_data['bu'].mean(), orig_data['bu'].std()
        sc_mean, sc_std = orig_data['sc'].mean(), orig_data['sc'].std()
        sod_mean, sod_std = orig_data['sod'].mean(), orig_data['sod'].std()

        # Validate standard deviations to avoid division by zero
        if bu_std == 0 or sc_std == 0 or sod_std == 0:
            logger.warning("Zero standard deviation detected in kidney function data")
            bu_std = max(bu_std, 1e-6)
            sc_std = max(sc_std, 1e-6)
            sod_std = max(sod_std, 1e-6)

        bu_z = (bu - bu_mean) / bu_std
        sc_z = (sc_orig - sc_mean) / sc_std
        sod_z = (sod - sod_mean) / sod_std

        kidney_score = bu_z + sc_z - sod_z
        engineered_features['kidney_func_score'] = kidney_score
        
        # 5. Symptom severity
        appet = 1 if df['appet'].iloc[0] == 'poor' else 0
        pe = 1 if df['pe'].iloc[0] == 'yes' else 0
        ane = 1 if df['ane'].iloc[0] == 'yes' else 0
        
        symptom_sev = appet + pe + ane
        engineered_features['symptom_severity'] = symptom_sev
        
        # Add required engineered features to the original dataframe
        df['eGFR'] = egfr
        df['comorb_score'] = comorb
        df['anemia_severity'] = anemia_sev
        df['symptom_severity'] = symptom_sev

        
        # Extract specified features for applying scaling and encoding:
        
        # Extract only the specified features
        feature_cols = ['anemia_severity','hemo','sg','comorb_score','eGFR','rbc','sc','dm','htn','bgr','symptom_severity']
        
        # Validate that all required features exist
        missing_feats = [col for col in feature_cols if col not in df.columns]
        if missing_feats:
            logger.error(f"Missing required features: {missing_feats}")
            logger.error(f"Available features: {list(df.columns)}")
            raise ValueError(f"Missing required features: {missing_feats}")
        
        # Extract the specified features
        df_features = df[feature_cols].copy()
        

        # Preprocessing:
        # Convert categorical columns to proper types
        cat_mapping = {
            'rbc': ['abnormal', 'missing', 'normal'], # This order is important, if not followed, the columns will not be encoded correctly
            'dm': ['no', 'yes'],
            'htn': ['no', 'yes']
        }
        
        for col, cats in cat_mapping.items():
            if col in df_features.columns:
                df_features[col] = pd.Categorical(df_features[col], categories=cats, ordered=False)
        
        
        # Apply log transformation to specified columns
        log_cols = ['bgr', 'sc', 'eGFR']
        for col in log_cols:
            if col in df_features.columns:
                df_features[col] = np.log(df_features[col])
        
        # Separate numerical and categorical columns for scaling
        num_cols_for_scaling = df_features.select_dtypes(include=['float64', 'int64', 'int32']).columns.tolist()
        cat_cols_for_encoding = df_features.select_dtypes(include=['category', 'object']).columns.tolist()
        
        # Get the feature names the scaler was trained on
        try:
            # Try to get feature names from scaler if available
            if hasattr(scaler, 'feature_names_in_'):
                scaler_features = list(scaler.feature_names_in_)
            else:
                # If not available, assume it was trained on all numerical features we have
                scaler_features = num_cols_for_scaling
                logger.info(f"Scaler feature names not available, using all numerical: {scaler_features}")
        except Exception as e:
            logger.warning(f"Could not determine scaler features: {e}")
            scaler_features = num_cols_for_scaling
        
        # Apply StandardScaler only to features it was trained on
        features_to_scale = [col for col in scaler_features if col in num_cols_for_scaling]
        if features_to_scale:
            df_features[features_to_scale] = scaler.transform(df_features[features_to_scale])
        else:
            logger.warning("No features to scale")
        
        # Apply one-hot encoding to categorical columns
        if cat_cols_for_encoding:
            df_features = pd.get_dummies(df_features, columns=cat_cols_for_encoding, drop_first=True)
        else:
            logger.info("No categorical features to encode")
        
        logger.info(f"Final dataframe columns: {list(df_features.columns)}")
        
        return df_features

    except Exception as e:
        logger.error(f"Error in preprocessing input: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        raise


@app.route('/')
def index():
    """Render the main application page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if model is None:
            return jsonify({
                'error': 'Model not loaded. Please check server configuration.',
                'success': False
            }), 500
        
        # Get form data
        form_data = request.form.to_dict()

        logger.info(f"Prediction request received at {datetime.now()}")

        features_df = preprocess_input(form_data)

        # Select only the features the model expects
        model_input_df = features_df[model.feature_names_]

        prediction = model.predict(model_input_df)[0]

        # Get prediction probability if available
        try:
            prediction_proba = model.predict_proba(model_input_df)[0]
            confidence = max(prediction_proba)
        except Exception as proba_error:
            logger.warning(f"Could not get prediction probabilities: {proba_error}")
            confidence = 0.85  # Default confidence if predict_proba not available

        # Determine result
        result = 'ckd' if prediction == 0 else 'notckd'

        # Prepare response with engineered features
        response = {
            'success': True,
            'prediction': result,
            'confidence': float(confidence),
            'engineered_features': engineered_features,
            'timestamp': datetime.now().isoformat(),
            'patient_data': patient_data
        }
        
        logger.info(f"Prediction completed: {result} (confidence: {confidence:.2f})")
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        # Only try to log these if the variables exist
        try:
            if 'features_df' in locals():
                logger.error(f"Input columns: {list(features_df.columns)}")
            if 'model' in globals() and model is not None and hasattr(model, 'feature_names_in_'):
                logger.error(f"Model feature_names_in_: {list(model.feature_names_in_)}")
        except Exception as logging_error:
            logger.error(f"Additional error during logging: {str(logging_error)}")
        
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'success': False
        }), 500


# Flask route for all chart data
@app.route('/get_chart_data/<chart_type>')
def get_chart_data(chart_type):
    import logging
    
    try:
        import pickle
        import pandas as pd
        import os
        
        # Check if data file exists
        data_file = 'data.pkl'
        if not os.path.exists(data_file):
            logger.error(f"[FLASK] Data file '{data_file}' not found!")
            return jsonify({'error': f'Data file {data_file} not found'}), 404
        
        # Load the dataframe
        with open(data_file, 'rb') as f:
            df = pickle.load(f)
        
        # Define required columns based on chart type
        chart_configs = {
            'sc_egfr': {
                'required_cols': ['sc', 'eGFR', 'classification'],
                'alt_names': {
                    'sc': ['serum_creatinine', 'creatinine', 'Serum_Creatinine'],
                    'eGFR': ['egfr', 'EGFR', 'estimated_gfr'],
                    'classification': ['target', 'label', 'class', 'Classification']
                }
            },
            'bu_anemia': {
                'required_cols': ['bu', 'anemia_severity', 'classification'],
                'alt_names': {
                    'bu': ['blood_urea', 'BU', 'Blood_Urea'],
                    'anemia_severity': ['anemia', 'Anemia_Severity', 'anemia_level'],
                    'classification': ['target', 'label', 'class', 'Classification']
                }
            },
            'pairplot': {
                'required_cols': ['sc', 'bu', 'hemo', 'classification'],
                'alt_names': {
                    'sc': ['serum_creatinine', 'creatinine', 'Serum_Creatinine'],
                    'bu': ['blood_urea', 'BU', 'Blood_Urea'],
                    'hemo': ['hemoglobin', 'Hemoglobin', 'Hemo'],
                    'classification': ['target', 'label', 'class', 'Classification']
                }
            }
        }
        
        # Validate chart type
        if chart_type not in chart_configs:
            logger.error(f"[FLASK] Invalid chart type: {chart_type}")
            return jsonify({'error': f'Invalid chart type: {chart_type}. Valid types: {list(chart_configs.keys())}'}), 400
        
        config = chart_configs[chart_type]
        required_cols = config['required_cols']
        alt_names = config['alt_names']
        
        # Check for required columns and map alternative names
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            for missing_col in missing_cols:
                found = False
                for alt_name in alt_names.get(missing_col, []):
                    if alt_name in df.columns:
                        df[missing_col] = df[alt_name]
                        found = True
                        break
                
                if not found:
                    logger.error(f"[FLASK] Required column '{missing_col}' not found. Available: {list(df.columns)}")
                    return jsonify({'error': f'Missing required column: {missing_col}'}), 400
        
        # Check for null values
        null_counts = df[required_cols].isnull().sum()
        if null_counts.any():
            logger.warning(f"[FLASK] Null values found: {null_counts.to_dict()}")
        
        # Prepare data for the chart
        chart_data = []
        
        for idx, row in df.iterrows():
            try:
                # Create data point based on chart type
                data_point = {}
                valid_point = True
                
                for col in required_cols[:-1]:  # Exclude classification
                    val = float(row[col]) if pd.notna(row[col]) else None
                    if val is None:
                        valid_point = False
                        break
                    data_point[col] = val
                
                if not valid_point:
                    continue
                
                # Add classification
                class_val = str(row['classification']) if pd.notna(row['classification']) else 'unknown'
                data_point['classification'] = class_val
                
                chart_data.append(data_point)
                
            except (ValueError, TypeError) as e:
                logger.warning(f"[FLASK] Skipping row {idx} due to data conversion error: {e}")
                continue
        
        # Log classification distribution
        class_dist = {}
        for item in chart_data:
            class_val = item['classification']
            class_dist[class_val] = class_dist.get(class_val, 0) + 1
        
        if len(chart_data) == 0:
            logger.error(f"[FLASK] No valid data points found after processing for {chart_type}!")
            return jsonify({'error': 'No valid data points found'}), 400
        
        return jsonify(chart_data)
        
    except FileNotFoundError as e:
        logger.error(f"[FLASK] File not found: {e}")
        return jsonify({'error': f'File not found: {str(e)}'}), 404
    except pd.errors.EmptyDataError as e:
        logger.error(f"[FLASK] Empty data file: {e}")
        return jsonify({'error': 'Data file is empty'}), 400
    except Exception as e:
        logger.error(f"[FLASK] Unexpected error: {str(e)}")
        logger.error(f"[FLASK] Error type: {type(e).__name__}")
        import traceback
        logger.error(f"[FLASK] Traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500



@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('index.html'), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        'error': 'Internal server error',
        'success': False
    }), 500


if __name__ == '__main__':
    # Load the model on startup
    if load_model():
        logger.info("Starting CKD Prediction Flask App")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        logger.error("Failed to load model. Exiting.")
        exit(1)
