import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

def calculate_metrics(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R2 Score): {r2:.4f}")

def predit(df: DataFrame):
    names = ['manufactured', 'power', 'no_of_owners', 'mileage', 'omv', 'arf', 'engine_cap', 'no_of_owners', 'depreciation', 'coe', 'road_tax', 'type_of_vehicle', 'category', 'transmission', 'eco_category', 'dereg_value', 'curb_weight', 'make']
    X = np.hstack((df[['manufactured']].values, df[['power']].values, df[['no_of_owners']].values, df[['mileage']].values, df[['omv']].values, df[['arf']].values, df[['engine_cap']].values, df[['no_of_owners']].values, df[['depreciation']].values, df[['coe']].values, df[['road_tax']].values, df[['type_of_vehicle']].values, df[['category']].values, df[['transmission']].values, df[['eco_category']].values, df[['dereg_value']].values, df[['curb_weight']].values, df[['make']].values), )
    y = df['price']
    # Create an imputer object with a mean filling strategy
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    # Fit on the training data and transform it
    X = imputer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def logistic_regression(X_train, X_test, y_train, y_test):
    lr_regressor = LogisticRegression(multi_class = 'multinomial',solver='newton-cg',max_iter = 1000).fit(X_train, y_train)
    # Predicting the Test set results
    Y_lr_predict = lr_regressor.predict(X_test)
    calculate_metrics(y_test, Y_lr_predict)

def random_forest(X_train, X_test, y_train, y_test):
    rf_regressor = RandomForestRegressor(n_estimators = 100, random_state = 42)
    rf_regressor.fit(X_train, y_train)
    # Predicting the Test set results
    Y_rf_predict = rf_regressor.predict(X_test)
    calculate_metrics(y_test, Y_rf_predict)
    # display featuere weight of random forest with feature names
    feature_importances = rf_regressor.feature_importances_
    sorted_idx = feature_importances.argsort()
    plt.barh(np.array(names)[sorted_idx], feature_importances[sorted_idx])
    plt.xlabel("Random Forest Feature Importance")
    plt.show()

def xgboost(X_train, X_test, y_train, y_test):
    xgb_regressor = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, random_state = 42)
    xgb_regressor.fit(X_train, y_train)
    # Predicting the Test set results
    Y_xgb_predict = xgb_regressor.predict(X_test)
    # calculate metrics
    calculate_metrics(y_test, Y_xgb_predict)

def MLP(X_train, X_test, y_train, y_test):
    mlp_regressor = MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=5000)
    mlp_regressor.fit(X_train, y_train)
    # predict
    Y_mlp_predict = mlp_regressor.predict(X_test)
    calculate_metrics(y_test, Y_mlp_predict)