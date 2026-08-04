from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from pathlib import Path
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

# ---------------------------------------------------------------------------
# Backend configuration
#
# The repo trains 3 feature-set "versions" (v0, v1, v2) x 2 tuning modes,
# saving each candidate model to PickleFiles/<Model_Name>_<version>_hyp.pkl
# (see Notebooks/4_ModelTraining.ipynb). Every model was fit on the exact
# feature set/encoding described by PickleFiles/ckd_<version>_artifacts.pkl.
#
# Pick MODEL_NAME to match a file in PickleFiles/, e.g. for
# "PickleFiles/Random_Forest_v2_hyp.pkl" set MODEL_NAME="Random_Forest" and
# VERSION_TAG="v2_hyp". ARTIFACT_VERSION is derived automatically.
# ---------------------------------------------------------------------------
MODEL_NAME = "Multi-layer_Perceptron"
VERSION_TAG = "v2_hyp"
ARTIFACT_VERSION = VERSION_TAG.split('_')[0]  # "v0" | "v1" | "v2"

BASE_DIR = Path(__file__).resolve().parent.parent
PICKLE_DIR = BASE_DIR / "PickleFiles"
DATASET_PATH = BASE_DIR / "Dataset" / "df_clean_eng.csv"

MODEL_PATH = PICKLE_DIR / f"{MODEL_NAME}_{VERSION_TAG}.pkl"
ARTIFACTS_PATH = PICKLE_DIR / f"ckd_{ARTIFACT_VERSION}_artifacts.pkl"

# Globals populated at startup
model = None
artifacts = None
training_data = None

# Raw fields collected from the prediction form (superset covering all versions)
NUM_COLS = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
CAT_COLS = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']


def load_model():
    """Load the trained model, its matching preprocessing artifacts, and the training data."""
    global model, artifacts, training_data
    try:
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
        logger.info(f"Model loaded successfully from {MODEL_PATH}")

        with open(ARTIFACTS_PATH, 'rb') as file:
            artifacts = pickle.load(file)
        logger.info(f"Artifacts loaded successfully from {ARTIFACTS_PATH}")

        training_data = pd.read_csv(DATASET_PATH)
        logger.info(f"Training data loaded successfully from {DATASET_PATH}")

        return True
    except FileNotFoundError as e:
        logger.error(f"Required file not found: {e}")
        return False
    except Exception as e:
        logger.error(f"Error loading model/artifacts: {str(e)}")
        return False


def preprocess_input(form_data):
    """
    Rebuild the engineered features from raw form data (mirrors
    Notebooks/3_FeatureEngineering.ipynb), then apply the log transform,
    scaler, and one-hot encoding that the selected model's version
    (ARTIFACT_VERSION) was trained with. Returns a DataFrame with exactly
    the columns/order the model expects.
    """
    patient_data = {}
    engineered_features = {}

    # Process numerical columns with validation
    for col in NUM_COLS:
        val = form_data.get(col, 0)
        try:
            patient_data[col] = float(val)
        except (ValueError, TypeError):
            logger.warning(f"Invalid numerical value for {col}: {val}, using 0")
            patient_data[col] = 0.0

    # Process categorical columns with validation
    for col in CAT_COLS:
        val = form_data.get(col, 'unknown')
        patient_data[col] = str(val) if val is not None else 'unknown'

    df = pd.DataFrame([patient_data])
    logger.info(f"Patient data: {patient_data}")

    # ---- Feature engineering (mirrors Notebooks/3_FeatureEngineering.ipynb) ----

    # 1. eGFR
    sc = max(df['sc'].iloc[0], 0.01)
    age = max(df['age'].iloc[0], 1)
    egfr = 186 * (sc ** (-1.154)) * (age ** (-0.203))
    engineered_features['eGFR'] = egfr

    # 2. Comorbidity score
    htn = 1 if df['htn'].iloc[0] == 'yes' else 0
    dm = 1 if df['dm'].iloc[0] == 'yes' else 0
    cad = 1 if df['cad'].iloc[0] == 'yes' else 0
    comorb = htn + dm + cad
    engineered_features['comorb_score'] = comorb

    # 3. Anemia severity, using z-scores from the training dataset
    hemo, pcv, rc = df['hemo'].iloc[0], df['pcv'].iloc[0], df['rc'].iloc[0]
    hemo_mean, hemo_std = training_data['hemo'].mean(), training_data['hemo'].std()
    pcv_mean, pcv_std = training_data['pcv'].mean(), training_data['pcv'].std()
    rc_mean, rc_std = training_data['rc'].mean(), training_data['rc'].std()
    hemo_std = max(hemo_std, 1e-6)
    pcv_std = max(pcv_std, 1e-6)
    rc_std = max(rc_std, 1e-6)
    hemo_z = (hemo - hemo_mean) / hemo_std
    pcv_z = (pcv - pcv_mean) / pcv_std
    rc_z = (rc - rc_mean) / rc_std
    anemia_sev = -(hemo_z + pcv_z + rc_z)
    engineered_features['anemia_severity'] = anemia_sev

    # 4. Kidney function score, using z-scores from the training dataset
    bu, sod = df['bu'].iloc[0], df['sod'].iloc[0]
    bu_mean, bu_std = training_data['bu'].mean(), training_data['bu'].std()
    sc_mean, sc_std = training_data['sc'].mean(), training_data['sc'].std()
    sod_mean, sod_std = training_data['sod'].mean(), training_data['sod'].std()
    bu_std = max(bu_std, 1e-6)
    sc_std = max(sc_std, 1e-6)
    sod_std = max(sod_std, 1e-6)
    bu_z = (bu - bu_mean) / bu_std
    sc_z = (sc - sc_mean) / sc_std
    sod_z = (sod - sod_mean) / sod_std
    kidney_score = bu_z + sc_z - sod_z
    engineered_features['kidney_func_score'] = kidney_score

    # 5. Symptom severity
    appet = 1 if df['appet'].iloc[0] == 'poor' else 0
    pe = 1 if df['pe'].iloc[0] == 'yes' else 0
    ane = 1 if df['ane'].iloc[0] == 'yes' else 0
    symptom_sev = appet + pe + ane
    engineered_features['symptom_severity'] = symptom_sev

    df['eGFR'] = egfr
    df['comorb_score'] = comorb
    df['anemia_severity'] = anemia_sev
    df['kidney_func_score'] = kidney_score
    df['symptom_severity'] = symptom_sev

    # ---- Apply the fitted artifacts for the selected version ----

    raw_feature_cols = artifacts['raw_feature_cols']
    missing_feats = [col for col in raw_feature_cols if col not in df.columns]
    if missing_feats:
        logger.error(f"Missing required features: {missing_feats}")
        raise ValueError(f"Missing required features: {missing_feats}")

    df_features = df[raw_feature_cols].copy()

    # Match training-time category sets/order so drop_first dummies line up
    for col, cats in artifacts['cat_categories'].items():
        if col in df_features.columns:
            df_features[col] = pd.Categorical(df_features[col], categories=cats, ordered=False)

    # Log-transform
    for col in artifacts['log_transform_cols']:
        if col in df_features.columns:
            df_features[col] = np.log(df_features[col])

    # Scale using the fitted scaler (only columns it was actually fit on)
    scaler = artifacts['scaler']
    scaler_features = list(getattr(scaler, 'feature_names_in_', artifacts['num_cols']))
    features_to_scale = [col for col in scaler_features if col in df_features.columns]
    if features_to_scale:
        df_features[features_to_scale] = scaler.transform(df_features[features_to_scale])

    # One-hot encode categorical columns
    cat_cols = artifacts['cat_cols']
    if cat_cols:
        df_features = pd.get_dummies(df_features, columns=cat_cols, drop_first=True)

    # Align to the exact column set/order the model was trained on
    df_features = df_features.reindex(columns=artifacts['feature_columns'], fill_value=0)

    logger.info(f"Final dataframe columns: {list(df_features.columns)}")

    return df_features, engineered_features, patient_data


@app.route('/')
def index():
    """Render the main application page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if model is None or artifacts is None:
            return jsonify({
                'error': 'Model not loaded. Please check server configuration.',
                'success': False
            }), 500

        form_data = request.form.to_dict()
        logger.info(f"Prediction request received at {datetime.now()}")

        model_input_df, engineered_features, patient_data = preprocess_input(form_data)

        prediction = model.predict(model_input_df)[0]
        result_label = artifacts['label_encoder'].inverse_transform([prediction])[0]
        result = 'ckd' if result_label == 'ckd' else 'notckd'

        try:
            prediction_proba = model.predict_proba(model_input_df)[0]
            confidence = max(prediction_proba)
        except Exception as proba_error:
            logger.warning(f"Could not get prediction probabilities: {proba_error}")
            confidence = 0.85

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
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'success': False
        }), 500


# Flask route for all chart data
@app.route('/get_chart_data/<chart_type>')
def get_chart_data(chart_type):
    try:
        if training_data is None:
            return jsonify({'error': f'Training data not loaded'}), 404

        df = training_data

        # Define required columns based on chart type
        chart_configs = {
            'sc_egfr': {
                'required_cols': ['sc', 'eGFR', 'classification'],
            },
            'bu_anemia': {
                'required_cols': ['bu', 'anemia_severity', 'classification'],
            },
            'pairplot': {
                'required_cols': ['sc', 'bu', 'hemo', 'classification'],
            }
        }

        if chart_type not in chart_configs:
            logger.error(f"[FLASK] Invalid chart type: {chart_type}")
            return jsonify({'error': f'Invalid chart type: {chart_type}. Valid types: {list(chart_configs.keys())}'}), 400

        required_cols = chart_configs[chart_type]['required_cols']

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"[FLASK] Required columns not found: {missing_cols}. Available: {list(df.columns)}")
            return jsonify({'error': f'Missing required columns: {missing_cols}'}), 400

        chart_data = []
        for idx, row in df.iterrows():
            try:
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

                data_point['classification'] = str(row['classification']) if pd.notna(row['classification']) else 'unknown'
                chart_data.append(data_point)

            except (ValueError, TypeError) as e:
                logger.warning(f"[FLASK] Skipping row {idx} due to data conversion error: {e}")
                continue

        if len(chart_data) == 0:
            logger.error(f"[FLASK] No valid data points found after processing for {chart_type}!")
            return jsonify({'error': 'No valid data points found'}), 400

        return jsonify(chart_data)

    except Exception as e:
        logger.error(f"[FLASK] Unexpected error: {str(e)}")
        import traceback
        logger.error(f"[FLASK] Traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_name': MODEL_NAME,
        'version_tag': VERSION_TAG,
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
    if load_model():
        logger.info(f"Starting CKD Prediction Flask App (model={MODEL_NAME}, version={VERSION_TAG})")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        logger.error("Failed to load model. Exiting.")
        exit(1)
