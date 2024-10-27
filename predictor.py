import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
# import xgboost as xgb
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

def predit(train_df, test_df):
    names = ['manufactured', 'power', 'no_of_owners', 'mileage', 'omv', 'arf', 'engine_cap', 'no_of_owners', 'depreciation', 'coe', 'road_tax', 'type_of_vehicle', 'category', 'transmission', 'eco_category', 'dereg_value', 'curb_weight', 'make']
    X_train = np.hstack((
        train_df[['manufactured']].values, 
        train_df[['power']].values, 
        train_df[['no_of_owners']].values, 
        train_df[['mileage']].values, 
        train_df[['omv']].values, 
        train_df[['arf']].values, 
        train_df[['engine_cap']].values, 
        train_df[['no_of_owners']].values, 
        train_df[['depreciation']].values, 
        train_df[['coe']].values, 
        # train_df[['road_tax']].values, 
        # train_df[['type_of_vehicle']].values, 
        # train_df[['category']].values, 
        # train_df[['transmission']].values, 
        # train_df[['eco_category']].values, 
        train_df[['dereg_value']].values, 
        # train_df[['curb_weight']].values, 
        # train_df[['make']].values
    ))

    X_test = train_df['price']

    # y_train = np.hstack((test_df[['manufactured']].values, test_df[['power']].values, test_df[['no_of_owners']].values, test_df[['mileage']].values, test_df[['omv']].values, test_df[['arf']].values, test_df[['engine_cap']].values, test_df[['no_of_owners']].values, test_df[['depreciation']].values, test_df[['coe']].values, test_df[['road_tax']].values, test_df[['type_of_vehicle']].values, test_df[['category']].values, test_df[['transmission']].values, test_df[['eco_category']].values, test_df[['dereg_value']].values, test_df[['curb_weight']].values, test_df[['make']].values), )
    y_train = np.hstack((test_df[['manufactured']].values, test_df[['power']].values, test_df[['no_of_owners']].values, test_df[['mileage']].values, test_df[['omv']].values, test_df[['arf']].values, test_df[['engine_cap']].values, test_df[['no_of_owners']].values, test_df[['depreciation']].values, test_df[['coe']].values,     test_df[['dereg_value']].values, ), )
    # y_test = test_df['price']
    y_test = 0
    # # Create an imputer object with a mean filling strategy
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    # # Fit on the training data and transform it
    X_train = imputer.fit_transform(X_train)
    y_train = imputer.fit_transform(y_train)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def logistic_regression(X_train, X_test, y_train, y_test):
    lr_regressor = LogisticRegression(multi_class = 'multinomial',solver='newton-cg',max_iter = 1000).fit(X_train, y_train)
    # Predicting the Test set results
    Y_lr_predict = lr_regressor.predict(X_test)
    calculate_metrics(y_test, Y_lr_predict)

def random_forest(X_train, X_test, y_train, y_test):
    rf_regressor = RandomForestRegressor(n_estimators = 100, random_state = 42)
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
    ## display featuere weight of random forest with feature names
    # feature_importances = rf_regressor.feature_importances_
    # sorted_idx = feature_importances.argsort()
    # plt.barh(np.array(names)[sorted_idx], feature_importances[sorted_idx])
    # plt.xlabel("Random Forest Feature Importance")
    # plt.show()

# def xgboost(X_train, X_test, y_train, y_test):
#     xgb_regressor = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, random_state = 42)
#     xgb_regressor.fit(X_train, y_train)
#     # Predicting the Test set results
#     Y_xgb_predict = xgb_regressor.predict(X_test)
#     # calculate metrics
#     calculate_metrics(y_test, Y_xgb_predict)

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