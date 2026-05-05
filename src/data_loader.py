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

def fetch_suite(suite_name='OpenML-CC18'):
    suite = openml.study.get_suite(suite_name)
    return suite


def load_dataset_info(task_id):
    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()
    
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=task.target_name
    )
    X = pd.DataFrame(X, columns=attribute_names)
    
    return {
        'task_id': task_id,
        'dataset_name': dataset.name,
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_categorical': sum(categorical_indicator),
        'n_numerical': X.shape[1] - sum(categorical_indicator),
        'n_classes': len(np.unique(y)),
        'has_missing': X.isnull().any().any()
    }


def load_all_datasets(suite):

    summary = []
    failed_tasks = []
    total = len(suite.tasks)
    
    for i, task_id in enumerate(suite.tasks):
        try:
            info = load_dataset_info(task_id)
            summary.append(info)
            print(f"[{i+1}/{total}] ✓ {info['dataset_name']} — "
                  f"{info['n_samples']} samples, "
                  f"{info['n_features']} features, "
                  f"{info['n_categorical']} categorical, "
                  f"{info['n_classes']} classes")
            
        except Exception as e:
            print(f"[{i+1}/{total}] ✗ Task {task_id} failed: {e}")
            failed_tasks.append(task_id)
    
    print(f"\n✓ Successfully loaded: {len(summary)} datasets")
    print(f"✗ Failed: {len(failed_tasks)} datasets")
    if failed_tasks:
        print(f"Failed task IDs: {failed_tasks}")
    
    return pd.DataFrame(summary), failed_tasks


def save_summary(summary_df, path=None):
    if path is None:
        path = os.path.expanduser(
            '~/Desktop/codes/EMPIRICAL PROJECT/tabular-benchmark/results/dataset_summary.csv'
        )

    path = os.path.expanduser(path)
    
    summary_df.to_csv(path, index=False)
    print(f"Summary saved to: {path}")


def print_summary_stats(summary_df):
    
    print("\n=== DATASET SUMMARY STATISTICS ===")
    print(f"Total datasets: {len(summary_df)}")
    print(f"\nSamples:")
    print(f"  Min:    {summary_df.n_samples.min()}")
    print(f"  Max:    {summary_df.n_samples.max()}")
    print(f"  Median: {summary_df.n_samples.median()}")
    print(f"\nFeatures:")
    print(f"  Min:    {summary_df.n_features.min()}")
    print(f"  Max:    {summary_df.n_features.max()}")
    print(f"  Median: {summary_df.n_features.median()}")
    print(f"\nTask types:")
    print(f"  Binary classification:     "
          f"{(summary_df.n_classes == 2).sum()}")
    print(f"  Multiclass classification: "
          f"{(summary_df.n_classes > 2).sum()}")
    print(f"\nDatasets with missing values: {summary_df.has_missing.sum()}")
    print(f"Datasets without missing values: "
          f"{(~summary_df.has_missing).sum()}")



if __name__ == "__main__":
    # Step 1: Fetch the suite
    suite = fetch_suite('OpenML-CC18')
    
    # Step 2: Load all datasets
    summary_df, failed_tasks = load_all_datasets(suite)
    
    # Step 3: Print statistics
    print_summary_stats(summary_df)
    
    # Step 4: Save results
    save_summary(summary_df)