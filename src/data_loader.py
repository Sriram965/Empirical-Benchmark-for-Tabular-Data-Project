import numpy as np
import openml
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
openml.config.apikey = os.getenv("OPENML_API_KEY")

# Set cache directory to project folder
openml.config.cache_directory = os.path.expanduser(
    '~/Desktop/codes/EMPIRICAL PROJECT/tabular-benchmark/data'
)

# ── Paper's 4 benchmark suites (Grinsztajn et al. 2022)
SUITES = {
    298: 'numerical_classification',
    297: 'numerical_regression',
    300: 'categorical_classification',
    299: 'categorical_regression',
}

def fetch_suite(suite_id):
    suite = openml.study.get_suite(suite_id)
    return suite


def load_dataset_info(task_id, suite_name):
    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()
    
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=task.target_name
    )
    X = pd.DataFrame(X, columns=attribute_names)

    is_regression = 'regression' in suite_name
    if is_regression:
        n_classes = None
    else:
        n_classes = len(np.unique(y))
    
    return {
        'task_id': task_id,
        'dataset_name': dataset.name,
        'suite_name':    suite_name,
        'task_type':     'regression' if is_regression else 'classification',
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_categorical': sum(categorical_indicator),
        'n_numerical': X.shape[1] - sum(categorical_indicator),
        'n_classes': n_classes,
        'has_missing': X.isnull().any().any()
    }


def load_all_datasets():
 
    summary = []
    failed_tasks = []
 
    for suite_id, suite_name in SUITES.items():
 
        suite = fetch_suite(suite_id)
        total = len(suite.tasks)
 
        for i, task_id in enumerate(suite.tasks):
            try:
                info = load_dataset_info(task_id, suite_name)
                summary.append(info)

                print(f"  [{i+1}/{total}] ✓ {info['dataset_name']} — "
                      f"{info['n_samples']} samples, "
                      f"{info['n_features']} features")
            except Exception as e:
                print(f"  [{i+1}/{total}] ✗ Task {task_id} failed: {e}")
                failed_tasks.append({'task_id': task_id, 'suite': suite_name})
 
    return pd.DataFrame(summary), failed_tasks


def save_summary(summary_df, path=None):
    if path is None:
        path = os.path.expanduser(
            '~/Desktop/codes/EMPIRICAL PROJECT/tabular-benchmark/results/dataset_summary.csv'
        )

    path = os.path.expanduser(path)
    
    summary_df.to_csv(path, index=False)




if __name__ == "__main__":
    
    # Step 1: Loading  all datasets
    summary_df, failed_tasks = load_all_datasets()

    
    # Step 2: Save results
    save_summary(summary_df)
