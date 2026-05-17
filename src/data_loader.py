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

# Paper's 4 benchmark suites (Grinsztajn et al. 2022)
SUITES = {
    298: 'numerical_classification',
    297: 'numerical_regression',
    300: 'categorical_classification',
    299: 'categorical_regression',
}

# This function fetches the suite information from OpenML given a suite_id. It uses the OpenML API to retrieve the suite details, which include the list of tasks associated with that suite. The function returns the suite object, which can then be used to access the tasks and other metadata related to the suite.
def fetch_suite(suite_id):
    suite = openml.study.get_suite(suite_id)
    return suite

#this loads the dataset info for a given task_id and suite_name, and returns a dictionary with relevant metadata about the dataset. It retrieves the task and dataset from OpenML, extracts the features and target variable, and computes various statistics such as the number of samples, features, categorical/numerical features, classes, and whether there are missing values.
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



# This function iterates through all the benchmark suites defined in the SUITES dictionary, loads the dataset information for each task in those suites, and compiles a summary of the datasets. It handles exceptions gracefully, logging any tasks that fail to load. The function returns a DataFrame containing the summary of all datasets and a list of any failed tasks.
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

# This function saves the summary DataFrame to a specified path as a CSV file. If no path is provided, it defaults to saving the file on the user's desktop under a specific directory. The function ensures that the path is expanded correctly and then writes the DataFrame to a CSV file without including the index.
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
    
    # Step 2: Saving all the meta results of the datasets into a csv file
    save_summary(summary_df)
