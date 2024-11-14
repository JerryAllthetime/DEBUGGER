import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from pandas import DataFrame


def calculate_metrics(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R2 Score): {r2:.4f}")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd


from sklearn.ensemble import RandomForestRegressor
import pandas as pd


def logistic_regression(X_train, X_test, y_train, y_test):
    lr_regressor = LogisticRegression(multi_class = 'multinomial',solver='newton-cg',max_iter = 1000).fit(X_train, y_train)
    # Predicting the Test set results
    Y_lr_predict = lr_regressor.predict(X_test)
    calculate_metrics(y_test, Y_lr_predict)

def random_forest(X_train, X_test, y_train, y_test):
    
    rf_regressor = RandomForestRegressor(n_estimators = 10000, random_state = 42)
    rf_regressor.fit(X_train, X_test)
    # Predicting the Test set results
    Y_rf_predict = rf_regressor.predict(y_train)
    ids = range(len(Y_rf_predict))


    submission_df = pd.DataFrame({
        'Id': ids,
        'Predicted': Y_rf_predict
    })


    submission_df.to_csv('submission.csv', index=False)
    calculate_metrics(y_test, Y_rf_predict)
    # display featuere weight of random forest with feature names
    feature_importances = rf_regressor.feature_importances_
    sorted_idx = feature_importances.argsort()
    # plt.barh(np.array(names)[sorted_idx], feature_importances[sorted_idx])
    # plt.xlabel("Random Forest Feature Importance")
    # plt.show()

def xgboost(X_train, X_test, y_train, y_test):
    xgb_regressor = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, random_state = 42)
    xgb_regressor.fit(X_train, y_train)
    # Predicting the Test set results
    Y_xgb_predict = xgb_regressor.predict(X_test)
    # calculate metrics
    calculate_metrics(y_test, Y_xgb_predict)

def MLP(X_train, X_test, y_train, y_test):
    mlp_regressor = MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=5000)
    # mlp_regressor.fit(X_train, y_train)
    mlp_regressor.fit(X_train, X_test)
    # predict
    Y_mlp_predict = mlp_regressor.predict(y_train)
    # calculate_metrics(y_test, Y_mlp_predict)
    ids = range(len(Y_mlp_predict))


    submission_df = pd.DataFrame({
        'Id': ids,
        'Predicted': Y_mlp_predict
    })


    submission_df.to_csv('submission.csv', index=False)
    calculate_metrics(y_test, Y_mlp_predict)