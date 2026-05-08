import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor



# Tree-based Models


#Random Forest Tree model
def get_random_forest(trial, task_type):
    
    params = {
        'n_estimators':     trial.suggest_int('n_estimators', 100, 500),
        'max_depth':        trial.suggest_int('max_depth', 3, 15),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features':     trial.suggest_float('max_features', 0.1, 1.0),
        'random_state':     42,
        'n_jobs':           -1 
    }

    if task_type == 'classification':
        return RandomForestClassifier(**params)
    return RandomForestRegressor(**params)



#HistGradientBoosting Tree model
def get_hist_gradient_boosting(trial, task_type):
    
    params = {
        'learning_rate':      trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'max_iter':           trial.suggest_int('max_iter', 100, 500),
        'max_depth':          trial.suggest_int('max_depth', 3, 10),
        'min_samples_leaf':   trial.suggest_int('min_samples_leaf', 1, 20),
        'l2_regularization':  trial.suggest_float('l2_regularization', 1e-6, 10.0, log=True),
        'random_state':       42
    }

    if task_type == 'classification':
        return HistGradientBoostingClassifier(**params)
    return HistGradientBoostingRegressor(**params)


def get_xgboost(trial, task_type):
    
    params = {
        'learning_rate':    trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'n_estimators':     trial.suggest_int('n_estimators', 100, 500),
        'max_depth':        trial.suggest_int('max_depth', 3, 10),
        'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_lambda':       trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'reg_alpha':        trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'random_state':     42,
        'n_jobs':           -1,
        'verbosity':        0
    }
    if task_type == 'classification':
        return XGBClassifier(**params, eval_metric='logloss',
                             use_label_encoder=False)
    return XGBRegressor(**params)