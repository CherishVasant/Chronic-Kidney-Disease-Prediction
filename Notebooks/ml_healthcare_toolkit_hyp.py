
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score, validation_curve
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                              f1_score, roc_auc_score, average_precision_score,
                              confusion_matrix, roc_curve, precision_recall_curve,
                              classification_report)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import pickle
import os 

def get_hyperparameter_grids_for_small_dataset():
    
    hyperparameter_grids = {

        # RANDOM FOREST 
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
            'params': {
                'n_estimators': [10, 25, 50, 100],
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [20, 40, 60, 80],
                'min_samples_leaf': [10, 20, 30, 40],
                'max_features': ['sqrt', 'log2', 0.3, 0.5],
                'max_samples': [0.6, 0.7, 0.8, 0.9],
                'min_impurity_decrease': [0.0, 0.01, 0.02, 0.05],
                'ccp_alpha': [0.0, 0.01, 0.02, 0.05]
            }
        },

        # EXTRA TREES
        'Extra Trees': {
            'model': ExtraTreesClassifier(random_state=42, class_weight='balanced'),
            'params': {
                'n_estimators': [10, 25, 50],
                'max_depth': [3, 5, 7],
                'min_samples_split': [40, 60, 80, 100],
                'min_samples_leaf': [20, 30, 40, 50],
                'max_features': ['sqrt', 'log2', 0.3],
                'max_samples': [0.6, 0.7, 0.8],
                'min_impurity_decrease': [0.01, 0.02, 0.05],
                'ccp_alpha': [0.01, 0.02, 0.05, 0.1]
            }
        },

        # DECISION TREE
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
            'params': {
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [40, 60, 80, 100],
                'min_samples_leaf': [20, 30, 40, 50],
                'max_features': ['sqrt', 'log2', 0.3, 0.5, None],
                'min_impurity_decrease': [0.01, 0.02, 0.05, 0.1],
                'ccp_alpha': [0.01, 0.02, 0.05, 0.1, 0.2]
            }
        },

        # GRADIENT BOOSTING
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [10, 25, 50, 100],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [2, 3, 4, 5],
                'min_samples_split': [40, 60, 80],
                'min_samples_leaf': [20, 30, 40],
                'subsample': [0.6, 0.7, 0.8, 0.9],
                'max_features': ['sqrt', 'log2', 0.3, 0.5],
                'min_impurity_decrease': [0.01, 0.02, 0.05],
                'ccp_alpha': [0.0, 0.01, 0.02]
            }
        },

        # XGBOOST 
        'XGBoost': {
            'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'params': {
                'n_estimators': [50, 100,150,200],
                'learning_rate': [0.1, 0.15, 0.2, 0.3],
                'max_depth': [5,6,7,8],
                'min_child_weight': [ 3,5,7, 10],
                'subsample': [0.7, 0.8, 0.9, 1],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
                'reg_alpha': [0, 0.01, 0.1, 1],
                'reg_lambda': [0.1, 1, 5, 10],
                'gamma': [0, 0.1, 0.5, 1],
                'scale_pos_weight': [1, 2, 3]  # Handle class imbalance
            }
        },

        # LIGHTGBM
        'LightGBM': {
            'model': lgb.LGBMClassifier(random_state=42, verbose=-1),
            'params': {
                'n_estimators': [10, 25, 50, 100],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [2, 3, 4, 5],
                'num_leaves': [7, 15, 31, 63],
                'min_child_samples': [10, 20, 30, 40],
                'min_split_gain': [0.01, 0.1, 0.5, 1],
                'subsample': [0.6, 0.7, 0.8, 0.9],
                'colsample_bytree': [0.3, 0.5, 0.7, 0.9],
                'reg_alpha': [0, 0.01, 0.1, 1],
                'reg_lambda': [0.1, 1, 5, 10],
                'min_child_weight': [0.001, 0.01, 0.1, 1],
                'class_weight': ['balanced', None]
            }
        },

        # CATBOOST
        'CatBoost': {
            'model': CatBoostClassifier(random_state=42, verbose=False),
            'params': {
                'iterations': [10, 25, 50, 100],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'depth': [2, 3, 4, 5],
                'min_data_in_leaf': [10, 20, 30, 40],
                'l2_leaf_reg': [1, 3, 5, 10, 20],
                'subsample': [0.6, 0.7, 0.8, 0.9],
                'colsample_bylevel': [0.3, 0.5, 0.7, 0.9],
                'border_count': [32, 64, 128],
                'bagging_temperature': [0, 0.5, 1],
                'class_weights': [[1, 1], [1, 2], [1, 3]]  # Handle imbalance
            }
        },

        # ADABOOST
        'AdaBoost': {
            'model': AdaBoostClassifier(random_state=42),
            'params': {
                'n_estimators': [10, 25, 50, 100],
                'learning_rate': [0.01, 0.1, 0.5, 1.0, 2.0],
                'algorithm': ['SAMME', 'SAMME.R']
            }
        },

        # BAGGING
        'Bagging': {
            'model': BaggingClassifier(random_state=42),
            'params': {
                'n_estimators': [10, 25, 50],
                'max_samples': [0.6, 0.7, 0.8, 0.9],
                'max_features': [0.3, 0.5, 0.7, 0.9],
                'bootstrap': [True, False],
                'bootstrap_features': [True, False]
            }
        },

        # LOGISTIC REGRESSION - Healthcare optimized
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
            'params': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga'],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                'fit_intercept': [True, False]
            }
        },

        # SGD CLASSIFIER
        'SGD Classifier': {
            'model': SGDClassifier(random_state=42, max_iter=1000, class_weight='balanced'),
            'params': {
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
                'eta0': [0.001, 0.01, 0.1, 1],
                'early_stopping': [True, False],
                'validation_fraction': [0.1, 0.2, 0.3]
            }
        },

        # SVM RBF - Healthcare optimized
        'SVM (RBF)': {
            'model': SVC(random_state=42, probability=True, class_weight='balanced'),
            'params': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf'],
                'shrinking': [True, False],
                'cache_size': [200, 500, 1000]
            }
        },

        # SVM LINEAR
        'SVM (Linear)': {
            'model': SVC(random_state=42, probability=True, class_weight='balanced'),
            'params': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'kernel': ['linear'],
                'shrinking': [True, False],
                'cache_size': [200, 500, 1000]
            }
        },

        # K-NEAREST NEIGHBORS
        'K-Nearest Neighbors': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7, 9, 11, 15, 21, 31],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'metric': ['euclidean', 'manhattan', 'minkowski'],
                'p': [1, 2, 3],
                'leaf_size': [10, 20, 30, 40, 50]
            }
        },

        # LINEAR DISCRIMINANT ANALYSIS
        'Linear Discriminant Analysis': {
            'model': LinearDiscriminantAnalysis(),
            'params': {
                'solver': ['svd', 'lsqr', 'eigen'],
                'shrinkage': [None, 'auto', 0.1, 0.3, 0.5, 0.7, 0.9],
                'priors': [None],
                'n_components': [None, 1, 2, 3]
            }
        },

        # QUADRATIC DISCRIMINANT ANALYSIS
        'Quadratic Discriminant Analysis': {
            'model': QuadraticDiscriminantAnalysis(),
            'params': {
                'reg_param': [0.0, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9],
                'store_covariance': [True, False],
                'tol': [1e-4, 1e-3, 1e-2]
            }
        },

        # MULTI-LAYER PERCEPTRON
        'Multi-layer Perceptron': {
            'model': MLPClassifier(random_state=42, max_iter=1000),
            'params': {
                'hidden_layer_sizes': [
                    (10,), (20,), (50,),
                    (10, 5), (20, 10), (50, 25),
                    (10, 10, 5), (20, 10, 5)
                ],
                'activation': ['tanh', 'relu', 'logistic'],
                'solver': ['adam', 'lbfgs'],
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
                'learning_rate': ['constant', 'invscaling', 'adaptive'],
                'learning_rate_init': [0.001, 0.01, 0.1],
                'early_stopping': [True, False],
                'validation_fraction': [0.1, 0.2, 0.3],
                'beta_1': [0.9, 0.95, 0.99],
                'beta_2': [0.999, 0.9999]
            }
        }
    }

    return hyperparameter_grids

def perform_healthcare_hyperparameter_tuning(X_train, y_train, model_name, scoring='recall', cv=5):
    
    grids = get_hyperparameter_grids_for_small_dataset()

    if model_name not in grids:
        print(f"Model {model_name} not found in hyperparameter grids")
        return None, None, None

    model_info = grids[model_name]
    model = model_info['model']
    param_grid = model_info['params']

    print(f"\nHEALTHCARE HYPERPARAMETER TUNING: {model_name}")
    print("-" * 70)
    print(f"   Primary Objective: Maximize Recall (Minimize False Negatives)")
    print(f"   Secondary Objective: Minimize Overfitting")
    print(f"   Cross-Validation Folds: {cv}")
    print(f"   Scoring Metric: {scoring}")

    param_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"   Parameter combinations to test: {param_combinations}")

    # Use RandomizedSearchCV for large parameter spaces to save computation time
    if param_combinations > 200:  # Increased threshold for better exploration
        n_iter = min(100, param_combinations)  # Test up to 100 combinations
        search = RandomizedSearchCV(
            model, param_grid, n_iter=n_iter, cv=cv, scoring=scoring,
            random_state=42, n_jobs=-1, verbose=0, return_train_score=True
        )
        search_type = f"RandomizedSearchCV (n_iter={n_iter})"
    else:
        search = GridSearchCV(
            model, param_grid, cv=cv, scoring=scoring,
            n_jobs=-1, verbose=0, return_train_score=True
        )
        search_type = "GridSearchCV (exhaustive)"

    print(f"   Search Strategy: {search_type}")

    # Fit the search
    print(f"   Training in progress...")
    search.fit(X_train, y_train)

    # Calculate overfitting metrics
    best_train_score = search.cv_results_['mean_train_score'][search.best_index_]
    best_val_score = search.best_score_
    overfitting_gap = best_train_score - best_val_score
    overfitting_ratio = overfitting_gap / best_train_score if best_train_score > 0 else 0

    print(f"\nHYPERPARAMETER TUNING RESULTS:")
    print(f"   Best {scoring.upper()} Score (CV): {best_val_score:.4f}")
    print(f"   Training Score: {best_train_score:.4f}")
    print(f"   Overfitting Gap: {overfitting_gap:.4f}")
    print(f"   Overfitting Ratio: {overfitting_ratio:.4f}")

    # Assess overfitting risk
    if overfitting_ratio > 0.15:
        overfitting_risk = "HIGH"
        risk_icon = "Risk"
    elif overfitting_ratio > 0.10:
        overfitting_risk = "MEDIUM"
        risk_icon = "Medium risk"
    else:
        overfitting_risk = "LOW"
        risk_icon = "Low risk"

    print(f"   {risk_icon} Overfitting Risk: {overfitting_risk}")

    print(f"\nBEST HYPERPARAMETERS:")
    for param, value in search.best_params_.items():
        print(f"  {param}: {value}")

    tuning_results = {
        'best_score': best_val_score,
        'train_score': best_train_score,
        'overfitting_gap': overfitting_gap,
        'overfitting_ratio': overfitting_ratio,
        'overfitting_risk': overfitting_risk,
        'search_type': search_type,
        'param_combinations_tested': n_iter if param_combinations > 200 else param_combinations
    }

    return search.best_estimator_, search.best_params_, tuning_results


def create_tuned_models_dict(X_train, y_train):
    
    print(f"\n{'='*80}")
    print("HEALTHCARE-OPTIMIZED HYPERPARAMETER TUNING")
    print("Optimizing for Minimal False Negatives in Kidney Disease Detection")
    print(f"{'='*80}")

    # Models to tune
    priority_models = [
        'Random Forest',
        'XGBoost',
        'LightGBM',
        'Logistic Regression',
        'SVM (RBF)',
        'Gradient Boosting',
        'CatBoost'
    ]

    secondary_models = [
        'Decision Tree',
        'Extra Trees',
        'K-Nearest Neighbors',
        'Linear Discriminant Analysis',
        'Multi-layer Perceptron',
        'AdaBoost',
        'SGD Classifier'
    ]

    tuned_models = {}
    tuning_summary = {}

   
    print(f"\nTUNING PRIORITY MODELS (Healthcare Critical):")
    print("=" * 60)

    for model_name in priority_models:
        try:
            best_model, best_params, tuning_results = perform_healthcare_hyperparameter_tuning(
                X_train, y_train, model_name, scoring='recall'  # Prioritize recall for healthcare
            )

            if best_model is not None:
                tuned_models[model_name] = best_model
                tuning_summary[model_name] = {
                    'params': best_params,
                    'results': tuning_results
                }
                print(f"  {model_name}: Successfully tuned")
            else:
                print(f"   {model_name}: Tuning failed")

        except Exception as e:
            print(f"   {model_name}: Error during tuning - {str(e)}")
            continue

    # Tune secondary models
    print(f"\nTUNING SECONDARY MODELS:")
    print("=" * 60)

    for model_name in secondary_models:
        try:
            best_model, best_params, tuning_results = perform_healthcare_hyperparameter_tuning(
                X_train, y_train, model_name, scoring='recall'
            )

            if best_model is not None:
                tuned_models[model_name] = best_model
                tuning_summary[model_name] = {
                    'params': best_params,
                    'results': tuning_results
                }
                print(f"   {model_name}: Successfully tuned")
            else:
                print(f"   {model_name}: Tuning failed")

        except Exception as e:
            print(f"   {model_name}: Error during tuning - {str(e)}")
            continue

    # Add baseline models 
    baseline_models = {
        'Gaussian Naive Bayes': GaussianNB(),
        'Multinomial Naive Bayes': MultinomialNB(),
        'Bernoulli Naive Bayes': BernoulliNB(),
        'Dummy Classifier (Stratified)': DummyClassifier(strategy='stratified', random_state=42),
        'Dummy Classifier (Most Frequent)': DummyClassifier(strategy='most_frequent', random_state=42)
    }

    tuned_models.update(baseline_models)

    print(f"\nHYPERPARAMETER TUNING SUMMARY:")
    print("=" * 60)
    print(f"   Priority Models Tuned: {len([m for m in priority_models if m in tuned_models])}/{len(priority_models)}")
    print(f"   Secondary Models Tuned: {len([m for m in secondary_models if m in tuned_models])}/{len(secondary_models)}")
    print(f"   Baseline Models Added: {len(baseline_models)}")
    print(f"   Total Models Ready: {len(tuned_models)}")

    return tuned_models, tuning_summary

def calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba):
    
    return {
        'Test Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'ROC AUC': roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) == 2 else 0,
        'PR AUC': average_precision_score(y_true, y_pred_proba) if len(np.unique(y_true)) == 2 else 0
    }

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

def plot_hyperparameter_tuning_summary(tuning_summary):
    """
    Create comprehensive visualization of hyperparameter tuning results
    """
    if not tuning_summary:
        print("No hyperparameter tuning results to plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Extract data
    model_names = list(tuning_summary.keys())
    scores = [info['results']['best_score'] for info in tuning_summary.values()]
    gaps = [info['results']['overfitting_gap'] for info in tuning_summary.values()]
    risks = [info['results']['overfitting_risk'] for info in tuning_summary.values()]

    # Color mapping for risks
    risk_colors = {'LOW': 'green', 'MEDIUM': 'orange', 'HIGH': 'red', 'CRITICAL': 'darkred'}
    colors = [risk_colors.get(risk, 'gray') for risk in risks]

    # 1. Recall Scores
    axes[0, 0].barh(range(len(model_names)), scores, color=colors, alpha=0.7)
    axes[0, 0].set_yticks(range(len(model_names)))
    axes[0, 0].set_yticklabels([name[:20] for name in model_names], fontsize=10)
    axes[0, 0].set_xlabel('Best CV Recall Score')
    axes[0, 0].set_title('Hyperparameter Tuning - Recall Scores')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Overfitting Gaps
    axes[0, 1].barh(range(len(model_names)), gaps, color=colors, alpha=0.7)
    axes[0, 1].set_yticks(range(len(model_names)))
    axes[0, 1].set_yticklabels([name[:20] for name in model_names], fontsize=10)
    axes[0, 1].set_xlabel('Overfitting Gap')
    axes[0, 1].set_title('Hyperparameter Tuning - Overfitting Gaps')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Risk Distribution
    risk_counts = pd.Series(risks).value_counts()
    axes[1, 0].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
                   colors=[risk_colors.get(risk, 'gray') for risk in risk_counts.index])
    axes[1, 0].set_title('Overfitting Risk Distribution (Tuned Models)')

    # 4. Score vs Gap scatter
    scatter = axes[1, 1].scatter(scores, gaps, c=colors, alpha=0.7, s=100, edgecolors='black')
    axes[1, 1].set_xlabel('Best CV Recall Score')
    axes[1, 1].set_ylabel('Overfitting Gap')
    axes[1, 1].set_title('Performance vs Overfitting (Tuned Models)')
    axes[1, 1].grid(True, alpha=0.3)

    # Add annotations for best models
    for i, (score, gap, name) in enumerate(zip(scores, gaps, model_names)):
        if score > np.percentile(scores, 75) and gap < np.percentile(gaps, 50):
            axes[1, 1].annotate(name[:15], (score, gap), xytext=(5, 5),
                              textcoords='offset points', fontsize=8)

    plt.tight_layout()
    plt.show()

    print(f"\nHYPERPARAMETER TUNING STATISTICS:")
    print(f"   Models Tuned: {len(tuning_summary)}")
    print(f"   Best Recall Score: {max(scores):.4f}")
    print(f"   Lowest Overfitting Gap: {min(gaps):.4f}")
    print(f"   Average Recall Improvement: {np.mean(scores):.4f}")
    print(f"   Average Overfitting Gap: {np.mean(gaps):.4f}")

def print_hyperparameter_tuning_summary(tuning_summary):
    
    print(f"\n{'='*80}")
    print("HYPERPARAMETER TUNING RESULTS SUMMARY")
    print(f"{'='*80}")

    if not tuning_summary:
        print("No hyperparameter tuning results available.")
        return

    # Sort by overfitting risk and best score
    risk_order = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}
    sorted_models = sorted(tuning_summary.items(),
                          key=lambda x: (risk_order.get(x[1]['results']['overfitting_risk'], 4),
                                       -x[1]['results']['best_score']))

    print(f"\nMODELS RANKED BY HEALTHCARE SUITABILITY:")
    print("   (Ordered by: Low Overfitting Risk → High Recall Score)")
    print("-" * 80)

    for i, (model_name, info) in enumerate(sorted_models, 1):
        results = info['results']
        risk_icon = {'LOW': 'low', 'MEDIUM': 'medium', 'HIGH': 'high'}.get(results['overfitting_risk'], 'risk')

        print(f"\n{i}. {model_name}")
        print(f"   {risk_icon} Overfitting Risk: {results['overfitting_risk']}")
        print(f"   Best Recall Score: {results['best_score']:.4f}")
        print(f"   Train-Val Gap: {results['overfitting_gap']:.4f}")
        print(f"   Search Method: {results['search_type']}")
        print(f"   Combinations Tested: {results['param_combinations_tested']}")

        # Show key hyperparameters 
        key_params = list(info['params'].items())[:5]  # Show first 5 params
        if key_params:
            print(f"   Key Parameters:")
            for param, value in key_params:
                print(f"   {param}: {value}")

        if len(info['params']) > 5:
            print(f" ...{len(info['params']) - 5} more parameters")

    # Risk distribution summary
    risk_counts = {}
    for model_name, info in tuning_summary.items():
        risk = info['results']['overfitting_risk']
        risk_counts[risk] = risk_counts.get(risk, 0) + 1

    print(f"\nOVERFITTING RISK DISTRIBUTION:")
    for risk, count in sorted(risk_counts.items(), key=lambda x: risk_order.get(x[0], 4)):
        icon = {'LOW': 'low', 'MEDIUM': 'medium', 'HIGH': 'high'}.get(risk, 'risk')
        percentage = count / len(tuning_summary) * 100
        print(f"   {icon} {risk}: {count} models ({percentage:.1f}%)")

    # Best performing models by metric
    best_recall = max(tuning_summary.items(), key=lambda x: x[1]['results']['best_score'])
    lowest_overfitting = min(tuning_summary.items(), key=lambda x: x[1]['results']['overfitting_gap'])

    print(f"\nSTANDOUT MODELS:")
    print(f"   Highest Recall: {best_recall[0]} ({best_recall[1]['results']['best_score']:.4f})")
    print(f"   Lowest Overfitting: {lowest_overfitting[0]} (Gap: {lowest_overfitting[1]['results']['overfitting_gap']:.4f})")

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
    """
    Plot comprehensive model comparison
    """
  
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

def save_trained_models_as_pickles(trained_models, version_tag="v2_hyp"):
    """
    Save each trained model as a separate pickle file
    """
    from pathlib import Path
    import pickle

    PROJECT_ROOT = Path.cwd().parent
    PICKLE_DIR = PROJECT_ROOT / "PickleFiles"
    PICKLE_DIR.mkdir(exist_ok=True)

    for model_name, model in trained_models.items():
        safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
        file_path = PICKLE_DIR / f"{safe_name}_{version_tag}.pkl"

        with open(file_path, "wb") as f:
            pickle.dump(model, f)

        print(f"Saved model → {file_path}")

def run_healthcare_ml_pipeline_hyp(X_train, X_test, y_train, y_test,version_tag="unknown"):

    print(f"\n{'='*80}")
    print("HEALTHCARE ML ANALYSIS PIPELINE")
    print("Kidney Disease Detection - Enhanced with Hyperparameter Tuning")
    print(f"{'='*80}")

    # Convert target to binary if needed
    if hasattr(y_train, 'dtype') and y_train.dtype == 'object':
        unique_classes = np.unique(y_train)
        if len(unique_classes) == 2:
            y_train_processed = (y_train == unique_classes[0]).astype(int)
            y_test_processed = (y_test == unique_classes[0]).astype(int)
        else:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_train_processed = le.fit_transform(y_train)
            y_test_processed = le.transform(y_test)
    else:
        y_train_processed = y_train
        y_test_processed = y_test

    print(f"   Dataset Info:")
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Classes: {len(np.unique(y_train_processed))}")
    print(f"   Class distribution: {dict(zip(*np.unique(y_train_processed, return_counts=True)))}")

    # Create hyperparameter-tuned models
    models, tuning_summary = create_tuned_models_dict(X_train, y_train_processed)
    print(f"\nTesting {len(models)} different classification algorithms...")
    print(f"   {len(tuning_summary)} models have been hyperparameter-tuned")
    print(f"   {len(models) - len(tuning_summary)} baseline models included")

    # Print hyperparameter tuning summary
    if tuning_summary:
        print_hyperparameter_tuning_summary(tuning_summary)

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
        print(f"\n ANALYZING: {model_name}")
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

            if model_name in tuning_summary:
                tuning_info = tuning_summary[model_name]
                print(f"\nHyperparameter Tuning Results:")
                print(f"  Best Recall Score (CV): {tuning_info['results']['best_score']:.4f}")
                print(f" Overfitting Risk: {tuning_info['results']['overfitting_risk']}")
                print(f" Overfitting Gap: {tuning_info['results']['overfitting_gap']:.4f}")
                print(f" Search Method: {tuning_info['results']['search_type']}")

                key_params = list(tuning_info['params'].items())[:3]  # Show first 3 params
                if key_params:
                    print(f"Key Tuned Parameters:")
                    for param, value in key_params:
                        print(f"{param}: {value}")

            
            print(f"\nGenerating Confusion Matrix...")
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

    if tuning_summary:
        print(f"\nGenerating Hyperparameter Tuning Summary Visualization...")
        plot_hyperparameter_tuning_summary(tuning_summary)

    print(f"\n{'='*80}")
    print(" FINAL HEALTHCARE RECOMMENDATIONS")
    print(f"{'='*80}")

    print(f"RECOMMENDED MODEL: {best_model_name}")
    best_stats = ranked_models.iloc[0]
    print(f"   Healthcare Score: {best_stats['Healthcare_Score']:.4f}")
    print(f"   Recall (Sensitivity): {best_stats['Recall']:.4f}")
    print(f"   Precision: {best_stats['Precision']:.4f}")
    print(f"   F1 Score: {best_stats['F1 Score']:.4f}")
    print(f"   Overfitting Risk: {best_stats['Overfitting Risk']}")

    if best_model_name in tuning_summary:
        best_tuning = tuning_summary[best_model_name]
        print(f"   Hyperparameter Optimization:")
        print(f"   Tuning Method: {best_tuning['results']['search_type']}")
        print(f"   CV Recall Score: {best_tuning['results']['best_score']:.4f}")
        print(f"   Overfitting Gap: {best_tuning['results']['overfitting_gap']:.4f}")

    print(f"\ TOPn 3 SAFE MODELS FOR HEALTHCARE:")
    safe_models = ranked_models[ranked_models['Overfitting Risk'].isin(['LOW', 'MEDIUM'])].head(3)
    for i, (_, row) in enumerate(safe_models.iterrows(), 1):
        risk_indicator = "Low" if row['Overfitting Risk'] == 'LOW' else "high"
        tuning_indicator = " " if row['Model'] in tuning_summary else " "
        print(f"   {i}. {tuning_indicator} {row['Model']} (Score: {row['Healthcare_Score']:.4f}, Risk: {risk_indicator}{row['Overfitting Risk']})")

    avoid_models = ranked_models[ranked_models['Overfitting Risk'].isin(['CRITICAL', 'HIGH'])]['Model'].tolist()
    if avoid_models:
        print(f"\nMODELS TO AVOID IN HEALTHCARE:")
        for model in avoid_models[:5]:  # Show top 5 to avoid
            if model in tuning_summary:
                gap = tuning_summary[model]['results']['overfitting_gap']
                print(f"    {model} (Overfitting Gap: {gap:.4f})")
            else:
                print(f"   {model}")

    if tuning_summary:
        print(f"\nHYPERPARAMETER TUNING INSIGHTS:")

        # Count tuned models by risk level
        tuned_risks = {}
        for model_name, info in tuning_summary.items():
            risk = info['results']['overfitting_risk']
            tuned_risks[risk] = tuned_risks.get(risk, 0) + 1

        print(f"  Tuned Models by Risk Level:")
        risk_order = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        for risk in risk_order:
            if risk in tuned_risks:
                icon = {'LOW': 'l', 'MEDIUM': 'm', 'HIGH': 'h', 'CRITICAL': 'c'}.get(risk, 'r')
                print(f"      {icon} {risk}: {tuned_risks[risk]} models")

        # Best tuning results
        best_tuned_recall = max(tuning_summary.items(), key=lambda x: x[1]['results']['best_score'])
        lowest_overfitting_tuned = min(tuning_summary.items(), key=lambda x: x[1]['results']['overfitting_gap'])

        print(f"   Best Tuned Recall: {best_tuned_recall[0]} ({best_tuned_recall[1]['results']['best_score']:.4f})")
        print(f"    Lowest Overfitting (Tuned): {lowest_overfitting_tuned[0]} (Gap: {lowest_overfitting_tuned[1]['results']['overfitting_gap']:.4f})")

    print(f"\nGENERATING ADDITIONAL ANALYSIS PLOTS...")

    # Overfitting risk distribution plot
    plt.figure(figsize=(15, 10))

    # Risk distribution
    plt.subplot(2, 3, 1)
    risk_counts = ranked_models['Overfitting Risk'].value_counts()
    colors = {'LOW': 'green', 'MEDIUM': 'orange', 'HIGH': 'red', 'CRITICAL': 'darkred'}
    risk_colors = [colors.get(risk, 'gray') for risk in risk_counts.index]
    plt.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', colors=risk_colors)
    plt.title('Overfitting Risk Distribution')

    # Healthcare scores distribution
    plt.subplot(2, 3, 2)
    plt.hist(ranked_models['Healthcare_Score'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Healthcare Score')
    plt.ylabel('Number of Models')
    plt.title('Healthcare Scores Distribution')
    plt.grid(True, alpha=0.3)

    # Recall vs Precision scatter plot
    plt.subplot(2, 3, 3)
    colors_risk = ranked_models['Overfitting Risk'].map(colors)
    scatter = plt.scatter(ranked_models['Recall'], ranked_models['Precision'],
                         c=colors_risk, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision')
    plt.title('Recall vs Precision (Colored by Risk)')
    plt.grid(True, alpha=0.3)

    # F1 Score vs CV Stability
    plt.subplot(2, 3, 4)
    plt.scatter(ranked_models['F1 Score'], ranked_models['CV Std F1'],
               c=colors_risk, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    plt.xlabel('F1 Score')
    plt.ylabel('CV Standard Deviation')
    plt.title('Performance vs Stability (Colored by Risk)')
    plt.grid(True, alpha=0.3)

    # Hyperparameter tuning comparison
    if tuning_summary:
        plt.subplot(2, 3, 5)
        tuned_models_data = []
        tuned_scores = []
        tuned_gaps = []
        for model_name in ranked_models['Model']:
            if model_name in tuning_summary:
                tuned_models_data.append(model_name[:15])  # Truncate long names
                tuned_scores.append(tuning_summary[model_name]['results']['best_score'])
                tuned_gaps.append(tuning_summary[model_name]['results']['overfitting_gap'])

        if tuned_models_data:
            plt.scatter(tuned_scores, tuned_gaps, alpha=0.7, s=60,
                       c='purple', edgecolors='black', linewidth=0.5)
            plt.xlabel('Tuned CV Recall Score')
            plt.ylabel('Overfitting Gap')
            plt.title('Hyperparameter Tuning Results')
            plt.grid(True, alpha=0.3)

            # Add model names as annotations for top performers
            for i, (score, gap, name) in enumerate(zip(tuned_scores, tuned_gaps, tuned_models_data)):
                if score > np.percentile(tuned_scores, 75) and gap < np.percentile(tuned_gaps, 50):
                    plt.annotate(name, (score, gap), xytext=(5, 5),
                               textcoords='offset points', fontsize=8)

    # Model complexity vs performance
    plt.subplot(2, 3, 6)
    # Create a complexity score based on model type
    complexity_map = {
        'Dummy': 1, 'Naive Bayes': 2, 'Logistic Regression': 3, 'LDA': 3, 'QDA': 4,
        'Decision Tree': 4, 'KNN': 4, 'SVM': 5, 'Random Forest': 6, 'Extra Trees': 6,
        'AdaBoost': 6, 'Gradient Boosting': 7, 'XGBoost': 8, 'LightGBM': 8, 'CatBoost': 8,
        'MLP': 9, 'SGD': 3, 'Ridge': 3, 'Bagging': 5
    }

    complexity_scores = []
    for model_name in ranked_models['Model']:
        complexity = 5  # default
        for key, value in complexity_map.items():
            if key.lower() in model_name.lower():
                complexity = value
                break
        complexity_scores.append(complexity)

    plt.scatter(complexity_scores, ranked_models['Healthcare_Score'],
               c=colors_risk, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    plt.xlabel('Model Complexity')
    plt.ylabel('Healthcare Score')
    plt.title('Complexity vs Healthcare Performance')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\n SUMMARY STATISTICS:")
    print(f"   Total Models Evaluated: {len(results_list)}")
    print(f"   Models with Hyperparameter Tuning: {len(tuning_summary)}")
    print(f"   Low Risk Models: {len(ranked_models[ranked_models['Overfitting Risk'] == 'LOW'])}")
    print(f"   Medium Risk Models: {len(ranked_models[ranked_models['Overfitting Risk'] == 'MEDIUM'])}")
    print(f"   High Risk Models: {len(ranked_models[ranked_models['Overfitting Risk'] == 'HIGH'])}")
    print(f"   Critical Risk Models: {len(ranked_models[ranked_models['Overfitting Risk'] == 'CRITICAL'])}")
    print(f"   Average Healthcare Score: {ranked_models['Healthcare_Score'].mean():.4f}")
    print(f"   Average Recall: {ranked_models['Recall'].mean():.4f}")
    print(f"   Average Precision: {ranked_models['Precision'].mean():.4f}")

    save_trained_models_as_pickles(trained_models, version_tag=version_tag)
    print(f"[INFO] Pickle files saved successfully for version: {version_tag}")

    return {
        'results': results_dict,
        'results_list': results_list,
        'cv_results': cv_results,
        'overfitting_summary': overfitting_summary,
        'risk_groups': risk_groups,
        'best_model': best_model_name,
        'ranked_models': ranked_models,
        'safe_models': safe_models,
        'trained_models': trained_models,
        'tuning_summary': tuning_summary,
        'hyperparameter_insights': {
            'tuned_models_count': len(tuning_summary),
            'best_tuned_recall': best_tuned_recall if tuning_summary else None,
            'lowest_overfitting_tuned': lowest_overfitting_tuned if tuning_summary else None,
            'risk_distribution': tuned_risks if tuning_summary else None
        }
    }


