from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
import numpy as np


def random_forest_cv(X_train, y_train):
    param_dist = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [None],
        'bootstrap': [True, False],
        'random_state': [42]
    }

    rf_regressor = RandomForestRegressor()
    random_search = RandomizedSearchCV(
        rf_regressor, param_distributions=param_dist, n_iter=100, cv=5,
        random_state=42, n_jobs=-1, verbose=2
    )
    random_search.fit(X_train, y_train)
    best_rf_regressor = random_search.best_estimator_
    print(f"Best Random Forest parameters: {random_search.best_params_}")

    cv_results = cross_val_score(best_rf_regressor, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    mean_rmse = np.sqrt(-cv_results.mean())
    print(f"Random Forest Cross-validated RMSE: {mean_rmse:.4f}")
    
    return best_rf_regressor

def logistic_regression_cv(X_train, y_train):
    param_dist = {
        'C': [0.001, 0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'saga'],
        'max_iter': [100, 200, 300],
        'penalty': ['l2']
    }

    log_regressor = LogisticRegression()
    random_search = RandomizedSearchCV(
        log_regressor, param_distributions=param_dist, n_iter=100, cv=5,
        random_state=42, n_jobs=-1, verbose=2
    )
    random_search.fit(X_train, y_train)
    best_log_regressor = random_search.best_estimator_
    print(f"Best Logistic Regression parameters: {random_search.best_params_}")

    cv_results = cross_val_score(best_log_regressor, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    mean_rmse = np.sqrt(-cv_results.mean())
    print(f"Logistic Regression Cross-validated RMSE: {mean_rmse:.4f}")

    return best_log_regressor



def xgboost_cv(X_train, y_train):
    param_dist = {
        'n_estimators': [100, 200, 500],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 2, 3],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    xgb_regressor = XGBRegressor()
    random_search = RandomizedSearchCV(
        xgb_regressor, param_distributions=param_dist, n_iter=100, cv=5,
        random_state=42, n_jobs=-1, verbose=2
    )
    random_search.fit(X_train, y_train)
    best_xgb_regressor = random_search.best_estimator_
    print(f"Best XGBoost parameters: {random_search.best_params_}")

    cv_results = cross_val_score(best_xgb_regressor, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    mean_rmse = np.sqrt(-cv_results.mean())
    print(f"XGBoost Cross-validated RMSE: {mean_rmse:.4f}")

    return best_xgb_regressor


def mlp_cv(X_train, y_train):
    param_dist = {
        'hidden_layer_sizes': [(50,), (100,), (200,)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam'],
        'alpha': [0.0001, 0.001, 0.01],
        'max_iter': [200, 300, 400]
    }

    mlp_regressor = MLPRegressor()
    random_search = RandomizedSearchCV(
        mlp_regressor, param_distributions=param_dist, n_iter=100, cv=5,
        random_state=42, n_jobs=-1, verbose=2
    )
    random_search.fit(X_train, y_train)
    best_mlp_regressor = random_search.best_estimator_
    print(f"Best MLP parameters: {random_search.best_params_}")

    cv_results = cross_val_score(best_mlp_regressor, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    mean_rmse = np.sqrt(-cv_results.mean())
    print(f"MLP Cross-validated RMSE: {mean_rmse:.4f}")

    return best_mlp_regressor
