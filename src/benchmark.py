import sys
import os
sys.path.append('..')

import numpy as np
import pandas as pd
import optuna
import time
import warnings
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, r2_score

from src.preprocessor import preprocess_dataset
from src.tree_models import get_random_forest, get_hist_gradient_boosting, get_xgboost
from src.dl_models import get_mlp, get_resnet, get_ft_transformer

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Configuration
N_TRIALS    = 20   
N_CV_FOLDS  = 3   
RESULTS_PATH = 'results/benchmark_results.csv'


# This maps model names to their factory functions.
TREE_MODELS = {
    'RandomForest':         get_random_forest,
    'HistGradientBoosting': get_hist_gradient_boosting,
    'XGBoost':              get_xgboost,
}

DL_MODELS = {
    'MLP':          get_mlp,
    'ResNet':       get_resnet,
    'FTTransformer':get_ft_transformer,
}


def get_cv_score(model, X_train, y_train, task_type):
    
    scoring = 'roc_auc' if task_type == 'classification' else 'r2'

    if task_type == 'classification':
        cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=42)

    scores = cross_val_score(model, X_train, y_train,
                             cv=cv, scoring=scoring, error_score='raise')
    return scores.mean()


def evaluate_model(model_name, model_fn, X_train, X_test,
                   y_train, y_test, task_type, input_dim, output_dim):
    
    def objective(trial):
        if model_name in TREE_MODELS:
            model = model_fn(trial, task_type)
        else:
            model = model_fn(trial, task_type, input_dim, output_dim)

        try:
            return get_cv_score(model, X_train, y_train, task_type)
        except Exception:
            return -1.0 if task_type == 'regression' else 0.0

    # Running the hyperparameter search
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    # Train the best configuration on the full training set
    best_trial = study.best_trial
    if model_name in TREE_MODELS:
        best_model = model_fn(best_trial, task_type)
    else:
        best_model = model_fn(best_trial, task_type, input_dim, output_dim)

    best_model.fit(X_train, y_train)

    # Evaluate once on the held-out test set
    if task_type == 'classification':
        # Use predict_proba for ROC-AUC — it needs probability scores
        # not hard class predictions
        if hasattr(best_model, 'predict_proba'):
            y_proba = best_model.predict_proba(X_test)[:, 1]
        else:
            y_proba = best_model.predict(X_test)
        test_score = roc_auc_score(y_test, y_proba)
    else:
        y_pred     = best_model.predict(X_test)
        test_score = r2_score(y_test, y_pred)

    return test_score, study.best_params


def run_benchmark(df):
    """
    Main benchmark loop — iterates over all datasets and all models.
    Saves results incrementally so a crash doesn't lose completed runs.
    """
    # Load existing results if the file already exists —
    # this lets you resume a benchmark that was interrupted
    if os.path.exists(RESULTS_PATH):
        results_df = pd.read_csv(RESULTS_PATH)
        completed  = set(zip(results_df['dataset_name'], results_df['model_name']))
        print(f"Resuming benchmark — {len(completed)} runs already completed.")
    else:
        results_df = pd.DataFrame()
        completed  = set()

    all_models = {**TREE_MODELS, **DL_MODELS}

    for _, row in df.iterrows():
        task_id    = int(row['task_id'])
        suite_name = row['suite_name']
        dataset    = row['dataset_name']
        task_type  = row['task_type']

        print(f"\n{'─'*55}")
        print(f"Dataset: {dataset} ({suite_name})")
        print(f"{'─'*55}")

        # Preprocess the dataset once and reuse for all 6 models
        try:
            X_train, X_test, y_train, y_test, _ = preprocess_dataset(
                task_id, suite_name
            )
        except Exception as e:
            print(f"  Preprocessing failed: {e} — skipping dataset")
            continue

        input_dim  = X_train.shape[1]
        output_dim = 2 if task_type == 'classification' else 1

        for model_name, model_fn in all_models.items():

            # Skip if this dataset-model combination already has a result
            if (dataset, model_name) in completed:
                print(f"  {model_name:<25} already done — skipping")
                continue

            print(f"  {model_name:<25} searching hyperparameters...")
            start_time = time.time()

            try:
                test_score, best_params = evaluate_model(
                    model_name, model_fn,
                    X_train, X_test, y_train, y_test,
                    task_type, input_dim, output_dim
                )
                elapsed = time.time() - start_time
                metric  = 'roc_auc' if task_type == 'classification' else 'r2'

                print(f"  {model_name:<25} {metric}: {test_score:.4f}  "
                      f"({elapsed:.0f}s)")

                # Save result immediately — don't wait until the end
                result = {
                    'dataset_name': dataset,
                    'suite_name':   suite_name,
                    'task_type':    task_type,
                    'model_name':   model_name,
                    'metric':       metric,
                    'test_score':   round(test_score, 4),
                    'training_time':round(elapsed, 1),
                    'best_params':  str(best_params)
                }
                results_df = pd.concat(
                    [results_df, pd.DataFrame([result])],
                    ignore_index=True
                )
                results_df.to_csv(RESULTS_PATH, index=False)

            except Exception as e:
                print(f"  {model_name:<25} failed: {e}")
                elapsed = time.time() - start_time

    print(f"\n{'='*55}")
    print(f"Benchmark complete. Results saved to {RESULTS_PATH}")
    print(f"Total runs completed: {len(results_df)}")
    return results_df