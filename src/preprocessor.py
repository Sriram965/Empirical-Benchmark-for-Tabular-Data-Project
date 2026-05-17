import numpy as np
import pandas as pd
import openml
import os
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, OrdinalEncoder, LabelEncoder


load_dotenv()
openml.config.apikey = os.getenv('OPENML_API_KEY')
openml.config.cache_directory = os.path.expanduser('~/.openml/cache')


#Sample Capping 
def cap_samples(X, y, max_samples=10000, random_state=42):
    if len(X) > max_samples:
        X_sampled = X.sample(n=max_samples, random_state=random_state)
        y_sampled = y.loc[X_sampled.index]
        return X_sampled.reset_index(drop=True), y_sampled.reset_index(drop=True)
    return X.reset_index(drop=True), y.reset_index(drop=True)



#Train-test split Function 
def split_data(X, y, test_size=0.2, random_state=42):
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )
    
    X_train = X_train.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)
    
    return X_train, X_test, y_train, y_test


#scaling numerical features using QuantileTransformer
def scale_numerical_features(X_train, X_test, categorical_indicator, attribute_names):
    
    # Identifing which columns are numerical
    num_cols = [col for col, is_cat in zip(attribute_names, categorical_indicator)
                if not is_cat]
    
    if len(num_cols) == 0:
        return X_train, X_test
    
    X_train_scaled = X_train.copy()
    X_test_scaled  = X_test.copy()
    
    # Fit the transformer on training data only
    qt = QuantileTransformer(output_distribution='normal', random_state=42)
    
    # Fit and transform training set, then transform test set using
    # the same fitted transformer — this is the critical step
    X_train_scaled[num_cols] = qt.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols]  = qt.transform(X_test[num_cols])
    
    return X_train_scaled, X_test_scaled


#Enoding categorical features using OrdinalEncoder
def encode_categorical_features(X_train, X_test, categorical_indicator, attribute_names):
    
    cat_cols = [col for col, is_cat in zip(attribute_names, categorical_indicator)
                if is_cat]
    
    if len(cat_cols) == 0:
        return X_train, X_test
    
    X_train_encoded = X_train.copy()
    X_test_encoded  = X_test.copy()
    
    # Fitting  on the training data only
    encoder = OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )
    
    X_train_encoded[cat_cols] = encoder.fit_transform(X_train[cat_cols])
    X_test_encoded[cat_cols]  = encoder.transform(X_test[cat_cols])
    
    return X_train_encoded, X_test_encoded


#Transforming the target variable

def transform_target(y_train, y_test, task_type):
    
    if task_type == 'regression':
        y_train_vals = y_train.astype(float).values.reshape(-1, 1)
        y_test_vals  = y_test.astype(float).values.reshape(-1, 1)
        
        qt = QuantileTransformer(output_distribution='normal', random_state=42)
        
        # Fitting  on training target only — never on the full y
        y_train_transformed = qt.fit_transform(y_train_vals).ravel()
        y_test_transformed  = qt.transform(y_test_vals).ravel()
        
        return y_train_transformed, y_test_transformed, qt
    
    else:
        le = LabelEncoder()
        y_train_transformed = le.fit_transform(y_train.astype(str))
        y_test_transformed  = le.transform(y_test.astype(str))
        
        return y_train_transformed, y_test_transformed, le


#Wrapping everything in a single function

def preprocess_dataset(task_id, suite_name):
    """
    Full preprocessing pipeline for a single dataset.
    Takes a task_id and suite_name, loads the raw data from OpenML,
    and returns model-ready train and test sets with all transformations
    applied in the correct order:

        1. Cap samples at 10,000
        2. Train/test split 80/20
        3. Scale numerical features with QuantileTransformer
        4. Encode categorical features with OrdinalEncoder
        5. Transform target variable

    Returns X_train, X_test, y_train, y_test, and the target transformer
    so predictions can be inverse-transformed back to the original scale
    for regression tasks.
    """
    # Step 1 — Loading the  raw data from OpenML
    task    = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=task.target_name
    )
    X = pd.DataFrame(X, columns=attribute_names)
    y = pd.Series(y)
    
    # Step 2 — Capping the  samples at 10,000
    X, y = cap_samples(X, y, max_samples=10000, random_state=42)
    
    # Step 3 — Split into train and test
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Step 4 — Scaling the  numerical features
    X_train, X_test = scale_numerical_features(
        X_train, X_test, categorical_indicator, attribute_names
    )
    
    # Step 5 — Encoding the  categorical features
    X_train, X_test = encode_categorical_features(
        X_train, X_test, categorical_indicator, attribute_names
    )
    
    # Step 6 — Transforming the  target variable
    task_type = 'regression' if 'regression' in suite_name else 'classification'
    y_train, y_test, target_transformer = transform_target(
        y_train, y_test, task_type
    )
    
    return X_train, X_test, y_train, y_test, target_transformer


