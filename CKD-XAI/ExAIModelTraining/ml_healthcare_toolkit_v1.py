import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, 
                               GradientBoostingClassifier, AdaBoostClassifier, 
                               BaggingClassifier)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                              f1_score, roc_auc_score, average_precision_score,
                              confusion_matrix, roc_curve, precision_recall_curve,
                              classification_report)
from sklearn.model_selection import StratifiedKFold, cross_val_score, validation_curve
from sklearn.preprocessing import LabelEncoder


def create_comprehensive_models_dict():

    models_dict = {
        # Linear Models
        'Logistic Regression (L1)': LogisticRegression(
            C=1.0, penalty='l1', solver='liblinear', random_state=42, max_iter=1000
        ),
        'Logistic Regression (L2)': LogisticRegression(
            C=1.0, penalty='l2', solver='liblinear', random_state=42, max_iter=1000
        ),
        'Logistic Regression (ElasticNet)': LogisticRegression(
            C=1.0, penalty='elasticnet', l1_ratio=0.5, solver='saga', random_state=42, max_iter=1000
        ),
        'Ridge Classifier': RidgeClassifier(alpha=1.0, random_state=42),
        'SGD Classifier': SGDClassifier(loss='log_loss', alpha=0.0001, random_state=42, max_iter=1000),

        # Tree-based Models
        'Decision Tree': DecisionTreeClassifier(
            max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=5,
            min_samples_leaf=2, random_state=42
        ),
        'Extra Trees': ExtraTreesClassifier(
            n_estimators=100, max_depth=10, min_samples_split=5,
            min_samples_leaf=2, random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=5,
            min_samples_split=5, min_samples_leaf=2, random_state=42
        ),
        'AdaBoost': AdaBoostClassifier(
            n_estimators=100, learning_rate=1.0, random_state=42
        ),
        'Bagging': BaggingClassifier(
            n_estimators=100, random_state=42
        ),

        # Advanced Gradient Boosting
        'XGBoost': XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, eval_metric='logloss'
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, verbose=-1
        ),
        'CatBoost': CatBoostClassifier(
            iterations=100, depth=6, learning_rate=0.1,
            random_state=42, verbose=False
        ),

        # Support Vector Machines
        'SVM (RBF)': SVC(
            C=1.0, kernel='rbf', probability=True, random_state=42
        ),
        'SVM (Linear)': SVC(
            C=1.0, kernel='linear', probability=True, random_state=42
        ),
        'SVM (Polynomial)': SVC(
            C=1.0, kernel='poly', degree=3, probability=True, random_state=42
        ),

        # Nearest Neighbors
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),

        # Naive Bayes
        'Gaussian Naive Bayes': GaussianNB(),
        'Multinomial Naive Bayes': MultinomialNB(),
        'Bernoulli Naive Bayes': BernoulliNB(),

        # Discriminant Analysis
        'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
        'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis(),

        # Neural Networks
        'Multi-layer Perceptron': MLPClassifier(
            hidden_layer_sizes=(100, 50), max_iter=500, random_state=42
        ),

        # Baseline
        'Dummy Classifier (Stratified)': DummyClassifier(
            strategy='stratified', random_state=42
        ),
        'Dummy Classifier (Most Frequent)': DummyClassifier(
            strategy='most_frequent', random_state=42
        )
    }

    return models_dict


def calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba):
    
    return {
        'Test Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'ROC AUC': roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) == 2 else 0,
        'PR AUC': average_precision_score(y_true, y_pred_proba) if len(np.unique(y_true)) == 2 else 0
    }


def plot_confusion_matrix(y_true, y_pred, model_name):
   
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


def plot_roc_curve(y_true, y_pred_proba, model_name):

    if len(np.unique(y_true)) == 2:  # Binary classification only
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.show()


def plot_precision_recall_curve(y_true, y_pred_proba, model_name):

    if len(np.unique(y_true)) == 2:  # Binary classification only
        plt.figure(figsize=(8, 6))
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = average_precision_score(y_true, y_pred_proba)
        plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend()
        plt.show()


def plot_feature_importance(model, X_train, model_name):

    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(10, 6))
        if hasattr(X_train, 'columns'):
            importances = pd.Series(model.feature_importances_, index=X_train.columns)
        else:
            importances = pd.Series(model.feature_importances_,
                                  index=[f'Feature_{i}' for i in range(len(model.feature_importances_))])
        importances.sort_values(ascending=False).head(10).plot(kind='bar')
        plt.title(f'Top 10 Feature Importances - {model_name}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


def enhanced_overfitting_detection(y_train, y_test, train_pred, test_pred, train_pred_proba, test_pred_proba, model_name):
    """
    Returns:
    - overfitting_score: 0-1 score where 1 is maximum overfitting
    - overfitting_risk: LOW/MEDIUM/HIGH/CRITICAL
    - overfitting_reasons: List of specific reasons
    """
    reasons = []
    overfitting_indicators = []

    # 1. Performance Gap Analysis
    train_f1 = f1_score(y_train, train_pred, average='weighted', zero_division=0)
    test_f1 = f1_score(y_test, test_pred, average='weighted', zero_division=0)
    f1_gap = train_f1 - test_f1

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    acc_gap = train_acc - test_acc

    # 2. Perfect Score Detection 
    perfect_train_score = (train_acc >= 0.999 or train_f1 >= 0.999)
    perfect_test_score = (test_acc >= 0.999 or test_f1 >= 0.999)

    # 3. Probability Distribution Analysis
    prob_variance_indicator = 0
    if hasattr(train_pred_proba, '__len__') and hasattr(test_pred_proba, '__len__'):
        try:
            # Check for extreme probability predictions
            train_extreme_probs = np.sum((train_pred_proba > 0.95) | (train_pred_proba < 0.05)) / len(train_pred_proba)
            test_extreme_probs = np.sum((test_pred_proba > 0.95) | (test_pred_proba < 0.05)) / len(test_pred_proba)

            # High confidence predictions that are wrong indicate overfitting
            if train_extreme_probs > 0.8:  # More than 80% extreme predictions
                prob_variance_indicator = 0.3
                reasons.append(f"High confidence predictions: {train_extreme_probs:.1%} of training predictions are >95% or <5% confident")
        except:
            pass

    # 4. Low Performance + High Gap = Learning Noise
    if test_f1 < 0.7 and f1_gap > 0.05:
        reasons.append(f"Low test performance ({test_f1:.3f}) with large gap ({f1_gap:.3f}) suggests learning noise")

    # Calculate overfitting score (0-1)
    gap_score = min(max(f1_gap, 0) / 0.2, 1.0)  # Normalize gap to 0-1 (0.2 = max reasonable gap)
    perfect_score_penalty = 0.8 if (perfect_train_score or perfect_test_score) else 0
    low_perf_high_gap_penalty = 0.6 if (test_f1 < 0.7 and f1_gap > 0.05) else 0

    overfitting_score = min(gap_score + perfect_score_penalty + low_perf_high_gap_penalty + prob_variance_indicator, 1.0)

    # Determine risk level
    if overfitting_score >= 0.8 or perfect_train_score or perfect_test_score:
        risk_level = "CRITICAL"
        if perfect_train_score:
            reasons.append("Perfect training score detected (99.9%+) - highly suspicious")
        if perfect_test_score:
            reasons.append("Perfect test score detected (99.9%+) - possible data leakage")
    elif overfitting_score >= 0.5:
        risk_level = "HIGH"
    elif overfitting_score >= 0.2:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # Add gap-based reasons
    if f1_gap > 0.1:
        reasons.append(f"Very large F1 gap: {f1_gap:.3f} (train: {train_f1:.3f}, test: {test_f1:.3f})")
    elif f1_gap > 0.05:
        reasons.append(f"Large F1 gap: {f1_gap:.3f} (train: {train_f1:.3f}, test: {test_f1:.3f})")

    if acc_gap > 0.1:
        reasons.append(f"Very large accuracy gap: {acc_gap:.3f}")
    elif acc_gap > 0.05:
        reasons.append(f"Large accuracy gap: {acc_gap:.3f}")

    return {
        'overfitting_score': overfitting_score,
        'overfitting_risk': risk_level,
        'overfitting_reasons': reasons,
        'f1_gap': f1_gap,
        'acc_gap': acc_gap,
        'train_f1': train_f1,
        'test_f1': test_f1,
        'train_acc': train_acc,
        'test_acc': test_acc
    }


def detect_overfitting_comprehensive_enhanced(X_train, X_test, y_train, y_test, models_dict):

    results = []
    cv_results = {}
    overfitting_summary = {}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models_dict.items():
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE ANALYSIS: {name}")
        print(f"{'='*60}")

        try:
            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            # Get prediction probabilities
            if hasattr(model, 'predict_proba'):
                train_pred_proba = model.predict_proba(X_train)[:, 1] if len(np.unique(y_test)) == 2 else model.predict_proba(X_train).max(axis=1)
                test_pred_proba = model.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) == 2 else model.predict_proba(X_test).max(axis=1)
            else:
                train_pred_proba = model.decision_function(X_train) if hasattr(model, 'decision_function') else train_pred
                test_pred_proba = model.decision_function(X_test) if hasattr(model, 'decision_function') else test_pred

            metrics = calculate_comprehensive_metrics(y_test, test_pred, test_pred_proba)

            overfitting_analysis = enhanced_overfitting_detection(
                y_train, y_test, train_pred, test_pred, train_pred_proba, test_pred_proba, name
            )

            print(f"PERFORMANCE METRICS:")
            for metric_name, value in metrics.items():
                print(f"   {metric_name}: {value:.4f}")

            print(f"\n ENHANCED OVERFITTING ANALYSIS:")
            print(f"   Train F1 Score: {overfitting_analysis['train_f1']:.4f}")
            print(f"   Test F1 Score: {overfitting_analysis['test_f1']:.4f}")
            print(f"   F1 Performance Gap: {overfitting_analysis['f1_gap']:.4f}")
            print(f"   Accuracy Gap: {overfitting_analysis['acc_gap']:.4f}")
            print(f"   Overfitting Score: {overfitting_analysis['overfitting_score']:.3f}")
            print(f"   Overfitting Risk: {overfitting_analysis['overfitting_risk']}")

            if overfitting_analysis['overfitting_reasons']:
                print(f"   Overfitting Indicators:")
                for reason in overfitting_analysis['overfitting_reasons']:
                    print(f"     • {reason}")

            overfitting_summary[name] = {
                'risk': overfitting_analysis['overfitting_risk'],
                'score': overfitting_analysis['overfitting_score'],
                'reasons': overfitting_analysis['overfitting_reasons']
            }

            print(f"\nSTRATIFIED K-FOLD CROSS VALIDATION:")
            cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1_weighted')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

            print(f"   CV Scores: {[f'{score:.4f}' for score in cv_scores]}")
            print(f"   CV Mean: {cv_mean:.4f}")
            print(f"   CV Std: {cv_std:.4f}")
            print(f"   CV Coefficient of Variation: {cv_std/cv_mean:.4f}")

            if cv_std > 0.05:
                print("   High CV variance suggests instability/overfitting")

            cv_results[name] = cv_scores

            print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
            print(classification_report(y_test, test_pred, zero_division=0))

            result_dict = {
                'Model': name,
                'Train F1': overfitting_analysis['train_f1'],
                'CV Mean F1': cv_mean,
                'CV Std F1': cv_std,
                'Overfitting Gap': overfitting_analysis['f1_gap'],
                'Overfitting Score': overfitting_analysis['overfitting_score'],
                'Overfitting Risk': overfitting_analysis['overfitting_risk']
            }
            result_dict.update(metrics)
            results.append(result_dict)

        except Exception as e:
            print(f"   Model {name} failed: {str(e)}")
            overfitting_summary[name] = {
                'risk': 'FAILED',
                'score': 1.0,
                'reasons': [f"Model failed to run: {str(e)}"]
            }
            continue
    return results, cv_results, overfitting_summary


def healthcare_model_selection_algorithm(results):
    """
        Priority: Recall > Precision > Stability > Low Overfitting > F1 > Accuracy
    """
    print(f"\n{'='*80}")
    print(f"HEALTHCARE MODEL SELECTION ALGORITHM")
    print(f"For Kidney Disease Detection - Minimizing False Negatives")
    print(f"{'='*80}")

    df = pd.DataFrame(results)

    # Healthcare-specific weights
    weights = {
        'Recall': 0.35,           # Highest priority - avoid missing kidney disease
        'Precision': 0.20,        # Important but secondary to recall
        'F1 Score': 0.15,         # Balanced metric
        'CV Mean F1': 0.15,       # Stability indicator
        'Test Accuracy': 0.10,    # Less important in healthcare
        'ROC AUC': 0.05,          # Additional metric
        'Overfitting_Penalty': -0.40,  # MUCH higher penalty for overfitting
        'Critical_Penalty': -0.60,     # Even higher penalty for critical overfitting
        'Stability_Bonus': 0.10        # Bonus for stable models
    }

    print("HEALTHCARE SELECTION CRITERIA:")
    print(" Recall (Sensitivity): 35% - Minimize false negatives")
    print(" Precision: 20% - Reduce false positives")
    print(" F1 Score: 15% - Balanced performance")
    print(" Cross-Validation Stability: 15% - Consistent performance")
    print(" Test Accuracy: 10% - Overall correctness")
    print("  ROC AUC: 5% - Additional validation")
    print("  Overfitting Penalty: -40% - Strong penalty for unreliable models")
    print("  Critical Overfitting Penalty: -60% - Severe penalty for suspicious models")
    print("  Stability Bonus: +10% - Reward for low variance")

    # Normalize metrics to 0-1 scale
    metrics_to_normalize = ['Test Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'PR AUC', 'CV Mean F1']
    df_normalized = df.copy()

    for metric in metrics_to_normalize:
        if metric in df.columns:
            min_val = df[metric].min()
            max_val = df[metric].max()
            if max_val > min_val:
                df_normalized[metric] = (df[metric] - min_val) / (max_val - min_val)
            else:
                df_normalized[metric] = 1.0

    # Calculate penalties and bonuses
    df_normalized['Overfitting_Penalty'] = df['Overfitting Score'].apply(
        lambda x: -min(x, 1.0)  # Direct penalty based on overfitting score
    )

    df_normalized['Critical_Penalty'] = df['Overfitting Risk'].apply(
        lambda x: -1.0 if x == 'CRITICAL' else (-0.5 if x == 'HIGH' else 0)
    )

    df_normalized['Stability_Bonus'] = df['CV Std F1'].apply(
        lambda x: max(0, (0.05 - x) / 0.05) if x <= 0.05 else 0  # Bonus for low std
    )

    # Calculate weighted score
    df_normalized['Healthcare_Score'] = 0
    for metric, weight in weights.items():
        if metric in df_normalized.columns:
            df_normalized['Healthcare_Score'] += df_normalized[metric] * weight

    
    df_sorted = df_normalized.sort_values('Healthcare_Score', ascending=False)

    print(f"\nTOP 5 MODELS FOR HEALTHCARE APPLICATION:")
    print("-" * 100)
    for i, (_, row) in enumerate(df_sorted.head().iterrows(), 1):
        print(f"{i}. {row['Model']}")
        print(f"   Healthcare Score: {row['Healthcare_Score']:.4f}")
        print(f"   Recall: {row['Recall']:.4f} | Precision: {row['Precision']:.4f} | F1: {row['F1 Score']:.4f}")
        print(f"   Overfitting Risk: {row['Overfitting Risk']} | Score: {row['Overfitting Score']:.3f}")
        print(f"   CV Stability: {row['CV Mean F1']:.4f} ± {row['CV Std F1']:.4f}")
        print()

    best_model = df_sorted.iloc[0]['Model']
    best_score = df_sorted.iloc[0]['Healthcare_Score']

    print(f"RECOMMENDED MODEL FOR KIDNEY DISEASE DETECTION: {best_model}")
    print(f"   Healthcare Score: {best_score:.4f}")

    
    critical_models = df[df['Overfitting Risk'] == 'CRITICAL']['Model'].tolist()
    if critical_models:
        print(f"\nMODELS TO AVOID (Critical Overfitting Risk):")
        for model in critical_models:
            print(f"   {model}")

    return best_model, df_sorted


def plot_model_comparison(results):
  
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('F1 Score', ascending=False)

    
    plt.figure(figsize=(15, 8))
    metrics = ['Test Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'PR AUC']
    x = np.arange(len(metrics))
    width = 0.15

    # Plot top 5 models
    top_models = results_df.head(5)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, (_, row) in enumerate(top_models.iterrows()):
        values = [row[metric] for metric in metrics]
        plt.bar(x + i*width, values, width, label=row['Model'], color=colors[i])

    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Top 5 Models Comparison')
    plt.xticks(x + width*2, metrics, rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # Plot CV scores box plot
    plt.figure(figsize=(15, 8))
    cv_data = []
    model_names = []
    for _, row in results_df.head(10).iterrows():
        cv_data.append([row['CV Mean F1']] * 5)  # Simplified for visualization
        model_names.append(row['Model'])

    plt.boxplot(cv_data, labels=model_names)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Cross-Validation F1 Score')
    plt.title('Cross-Validation Performance Comparison (Top 10 Models)')
    plt.tight_layout()
    plt.show()


def validation_curve_analysis_enhanced(X_train, y_train, model, param_name, param_range):
    
    # Handle None values in parameter range for plotting
    plot_param_range = []
    plot_labels = []

    for param in param_range:
        if param is None:
            plot_param_range.append(999)  # Use large number for None
            plot_labels.append('None')
        else:
            plot_param_range.append(param)
            plot_labels.append(str(param))

    try:
        train_scores, val_scores = validation_curve(
            model, X_train, y_train, param_name=param_name,
            param_range=param_range, cv=5, scoring='f1_weighted'
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(plot_param_range, train_mean, 'o-', label='Training Score')
        plt.plot(plot_param_range, val_mean, 'o-', label='Validation Score')
        plt.fill_between(plot_param_range, train_mean - train_std, train_mean + train_std, alpha=0.1)
        plt.fill_between(plot_param_range, val_mean - val_std, val_mean + val_std, alpha=0.1)

        # Set custom x-tick labels
        plt.xticks(plot_param_range, plot_labels)
        plt.xlabel(param_name)
        plt.ylabel('F1 Score')
        plt.title(f'Validation Curve - {param_name}')
        plt.legend()
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f"Validation curve analysis failed: {str(e)}")


def data_leakage_detector(X_train, X_test, y_train, y_test):

    print(f"\n{'='*50}")
    print("DATA LEAKAGE DETECTION")
    print(f"{'='*50}")

    # Check for identical samples
    if hasattr(X_train, 'values'):
        X_train_vals = X_train.values
        X_test_vals = X_test.values
    else:
        X_train_vals = X_train
        X_test_vals = X_test

    # Check for duplicate rows between train and test
    train_set = set([tuple(row) for row in X_train_vals])
    test_set = set([tuple(row) for row in X_test_vals])
    overlap = train_set.intersection(test_set)

    print(f"Identical samples between train/test: {len(overlap)}")
    if len(overlap) > 0:
        print("CRITICAL: Data leakage detected - identical samples in train/test!")
    else:
        print("No identical samples found between train/test sets")

    # Check feature correlation with target
    if hasattr(X_train, 'corrwith'):
        correlations = X_train.corrwith(pd.Series(y_train))
        high_corr = correlations[abs(correlations) > 0.9]
        if len(high_corr) > 0:
            print(f"Features with suspiciously high correlation (>0.9): {high_corr.to_dict()}")
        else:
            print("No suspiciously high feature-target correlations found")

    return len(overlap)


def print_overfitting_summary(overfitting_summary):

    print(f"\n{'='*80}")
    print("OVERFITTING RISK SUMMARY")
    print(f"{'='*80}")

    risk_groups = {'CRITICAL': [], 'HIGH': [], 'MEDIUM': [], 'LOW': [], 'FAILED': []}

    for model, info in overfitting_summary.items():
        risk_groups[info['risk']].append((model, info))

    for risk_level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'FAILED']:
        models = risk_groups[risk_level]
        if models:
            icon = {'CRITICAL': 'Critical', 'HIGH': 'High', 'MEDIUM': 'Medium', 'LOW': 'Low', 'FAILED': 'Failed'}[risk_level]
            print(f"\n{icon} {risk_level} OVERFITTING RISK ({len(models)} models):")
            print("-" * 60)

            for model, info in models:
                print(f"   • {model}")
                print(f"     Score: {info['score']:.3f}")
                if info['reasons']:
                    for reason in info['reasons'][:2]:  # Show first 2 reasons
                        print(f"     - {reason}")
                print()

    # Summary statistics
    total_models = len(overfitting_summary)
    safe_models = len(risk_groups['LOW'])
    risky_models = len(risk_groups['HIGH']) + len(risk_groups['CRITICAL'])

    print(f"  SUMMARY STATISTICS:")
    print(f"   Total Models Evaluated: {total_models}")
    print(f"   Safe Models (LOW risk): {safe_models} ({safe_models/total_models*100:.1f}%)")
    print(f"   Risky Models (HIGH/CRITICAL): {risky_models} ({risky_models/total_models*100:.1f}%)")
    print(f"   Failed Models: {len(risk_groups['FAILED'])}")

    return risk_groups


def run_healthcare_ml_pipeline(X_train, X_test, y_train, y_test):
    
    print(f"\n{'='*80}")
    print("HEALTHCARE ML ANALYSIS PIPELINE")
    print("Kidney Disease Detection - Enhanced Overfitting Detection")
    print(f"{'='*80}")

    # Convert target to binary if needed
    if hasattr(y_train, 'dtype') and y_train.dtype == 'object':
        unique_classes = np.unique(y_train)
        if len(unique_classes) == 2:
            y_train_processed = (y_train == unique_classes[0]).astype(int)
            y_test_processed = (y_test == unique_classes[0]).astype(int)
        else:
            le = LabelEncoder()
            y_train_processed = le.fit_transform(y_train)
            y_test_processed = le.transform(y_test)
    else:
        y_train_processed = y_train
        y_test_processed = y_test

    print(f" Dataset Info:")
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Classes: {len(np.unique(y_train_processed))}")
    print(f"   Class distribution: {dict(zip(*np.unique(y_train_processed, return_counts=True)))}")

    # Create models dictionary
    models = create_comprehensive_models_dict()
    print(f"\nTesting {len(models)} different classification algorithms...")

    # Run comprehensive analysis with enhanced overfitting detection
    results_list, cv_results, overfitting_summary = detect_overfitting_comprehensive_enhanced(
        X_train, X_test, y_train_processed, y_test_processed, models
    )

    results_dict = {}
    for result in results_list:
        model_name = result['Model']
        results_dict[model_name] = result

    print(f"\n{'='*80}")
    print("INDIVIDUAL MODEL ANALYSIS WITH VISUALIZATIONS")
    print(f"{'='*80}")

    trained_models = {}

    # Individual model analysis with plotting
    for model_name, model_results in results_dict.items():
        if model_results is None:
            continue

        print(f"\nANALYZING: {model_name}")
        print("-" * 60)

        try:
            # Get model instance
            if model_name not in models:
                print(f"Model {model_name} not found in models dictionary, skipping...")
                continue

            model = models[model_name]

            
            model.fit(X_train, y_train_processed)
            trained_models[model_name] = model

            
            y_pred = model.predict(X_test)

            # Get prediction probabilities if available
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
                if len(np.unique(y_test_processed)) == 2:
                    y_pred_proba = y_pred_proba[:, 1]
                else:
                    y_pred_proba = y_pred_proba.max(axis=1)
            elif hasattr(model, 'decision_function'):
                y_pred_proba = model.decision_function(X_test)
                # Normalize decision function scores to [0,1] for binary classification
                if len(np.unique(y_test_processed)) == 2:
                    y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())
            else:
                y_pred_proba = None

            print(f"Model Performance:")
            
            print(f"   Test Accuracy: {model_results.get('Test Accuracy', 0):.4f}")
            print(f"   Precision: {model_results.get('Precision', 0):.4f}")
            print(f"   Recall: {model_results.get('Recall', 0):.4f}")
            print(f"   F1 Score: {model_results.get('F1 Score', 0):.4f}")
            print(f"   ROC AUC: {model_results.get('ROC AUC', 0):.4f}")

           
            print(f"\n Generating Confusion Matrix...")
            plot_confusion_matrix(y_test_processed, y_pred, model_name)

            
            if len(np.unique(y_test_processed)) == 2 and y_pred_proba is not None:
                print(f"Generating ROC Curve...")
                plot_roc_curve(y_test_processed, y_pred_proba, model_name)

                print(f"Generating Precision-Recall Curve...")
                plot_precision_recall_curve(y_test_processed, y_pred_proba, model_name)

            
            if hasattr(model, 'feature_importances_'):
                print(f"Generating Feature Importance Plot...")
                plot_feature_importance(model, X_train, model_name)
            elif hasattr(model, 'coef_') and model.coef_.ndim == 1:
                print(f"Generating Feature Coefficients Plot...")
                # Handle linear model coefficients
                plt.figure(figsize=(10, 6))
                if hasattr(X_train, 'columns'):
                    coef_series = pd.Series(np.abs(model.coef_), index=X_train.columns)
                else:
                    coef_series = pd.Series(np.abs(model.coef_), index=[f'Feature_{i}' for i in range(len(model.coef_))])
                coef_series.sort_values(ascending=False).head(10).plot(kind='bar')
                plt.title(f'Top 10 Feature Coefficients (Absolute) - {model_name}')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.show()

            print(f"Generating Validation Curve Analysis...")
            if model_name == 'Random Forest':
                validation_curve_analysis_enhanced(
                    X_train, y_train_processed, model,
                    'n_estimators', [10, 50, 100, 200, 300]
                )
            elif model_name == 'XGBoost':
                validation_curve_analysis_enhanced(
                    X_train, y_train_processed, model,
                    'max_depth', [3, 4, 5, 6, 7, 8]
                )
            elif model_name == 'LightGBM':
                validation_curve_analysis_enhanced(
                    X_train, y_train_processed, model,
                    'num_leaves', [10, 20, 30, 40, 50]
                )
            elif 'SVM' in model_name:
                validation_curve_analysis_enhanced(
                    X_train, y_train_processed, model,
                    'C', [0.1, 1, 10, 100, 1000]
                )
            elif 'Logistic Regression' in model_name:
                validation_curve_analysis_enhanced(
                    X_train, y_train_processed, model,
                    'C', [0.01, 0.1, 1, 10, 100]
                )

            print(f"Completed analysis for {model_name}\n")

        except Exception as e:
            print(f"Error analyzing {model_name}: {str(e)}")
            continue

    
    risk_groups = print_overfitting_summary(overfitting_summary)

    results_df = pd.DataFrame(results_list)

   
    best_model_name, ranked_models = healthcare_model_selection_algorithm(results_df)

    print(f"\n{'='*80}")
    print("COMPREHENSIVE MODEL COMPARISON VISUALIZATIONS")
    print(f"{'='*80}")

    print("Generating Top Models Comparison...")
    plot_model_comparison(results_df)

    print(f"\n{'='*80}")
    print("FINAL HEALTHCARE RECOMMENDATIONS")
    print(f"{'='*80}")

    print(f"RECOMMENDED MODEL: {best_model_name}")
    best_stats = ranked_models.iloc[0]
    print(f"   Healthcare Score: {best_stats['Healthcare_Score']:.4f}")
    print(f"   Recall (Sensitivity): {best_stats['Recall']:.4f}")
    print(f"   Precision: {best_stats['Precision']:.4f}")
    print(f"   F1 Score: {best_stats['F1 Score']:.4f}")
    print(f"   Overfitting Risk: {best_stats['Overfitting Risk']}")

    print(f"\nTOP 3 SAFE MODELS FOR HEALTHCARE:")
    safe_models = ranked_models[ranked_models['Overfitting Risk'].isin(['LOW', 'MEDIUM'])].head(3)
    for i, (_, row) in enumerate(safe_models.iterrows(), 1):
        print(f"   {i}. {row['Model']} (Score: {row['Healthcare_Score']:.4f}, Risk: {row['Overfitting Risk']})")

    # Models to avoid
    avoid_models = ranked_models[ranked_models['Overfitting Risk'].isin(['CRITICAL', 'HIGH'])]['Model'].tolist()
    if avoid_models:
        print(f"\nMODELS TO AVOID IN HEALTHCARE:")
        for model in avoid_models[:5]:  # Show top 5 to avoid
            print(f"  {model}")

    print(f"\n GENERATING ADDITIONAL ANALYSIS PLOTS...")

    # Overfitting risk distribution plot
    plt.figure(figsize=(12, 8))
    risk_counts = ranked_models['Overfitting Risk'].value_counts()
    colors = {'LOW': 'green', 'MEDIUM': 'orange', 'HIGH': 'red', 'CRITICAL': 'darkred'}
    risk_colors = [colors.get(risk, 'gray') for risk in risk_counts.index]

    plt.subplot(2, 2, 1)
    plt.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', colors=risk_colors)
    plt.title('Overfitting Risk Distribution')

    # Healthcare scores distribution
    plt.subplot(2, 2, 2)
    plt.hist(ranked_models['Healthcare_Score'], bins=15, alpha=0.7, color='skyblue')
    plt.xlabel('Healthcare Score')
    plt.ylabel('Number of Models')
    plt.title('Healthcare Scores Distribution')

    # Recall vs Precision scatter plot
    plt.subplot(2, 2, 3)
    colors_risk = ranked_models['Overfitting Risk'].map(colors)
    plt.scatter(ranked_models['Recall'], ranked_models['Precision'], c=colors_risk, alpha=0.7)
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision')
    plt.title('Recall vs Precision (Colored by Risk)')

    # F1 Score vs CV Stability
    plt.subplot(2, 2, 4)
    plt.scatter(ranked_models['F1 Score'], ranked_models['CV Std F1'], c=colors_risk, alpha=0.7)
    plt.xlabel('F1 Score')
    plt.ylabel('CV Standard Deviation')
    plt.title('Performance vs Stability (Colored by Risk)')

    plt.tight_layout()
    plt.show()

    return {
        'results': results_dict,
        'results_list': results_list,
        'cv_results': cv_results,
        'overfitting_summary': overfitting_summary,
        'risk_groups': risk_groups,
        'best_model': best_model_name,
        'ranked_models': ranked_models,
        'safe_models': safe_models,
        'trained_models': trained_models
    }
