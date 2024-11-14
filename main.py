import pandas as pd
import numpy as np
import argparse
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import numpy as np
from itertools import product
from sklearn.model_selection import cross_val_predict
import joblib
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
import category_encoders as ce
from EDA import *
from preprocessing import *
from predictor import *

def rmse_scorer(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def random_forest_sweep(X_train, y_train):
    # Split the data into training and test sets (80% train, 20% test)
    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Define the hyperparameter grid for RandomForest
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5,10],
        'min_samples_leaf': [1, 2,3]
    }
    
    # List to store the RMSE for each parameter setting
    all_rmse_results = []
    
    # Loop over all parameter combinations
    for params in product(*rf_param_grid.values()):
        rf_params = dict(zip(rf_param_grid.keys(), params))
        rf_model = RandomForestRegressor(random_state=42, **rf_params)
        
        # Train the model
        rf_model.fit(X_train_split, y_train_split)
        
        # Make predictions on the test set
        rf_pred = rf_model.predict(X_test_split)
        
        # Calculate RMSE
        rf_rmse = rmse_scorer(y_test_split, rf_pred)
        
        # Store the parameter combination and corresponding RMSE
        all_rmse_results.append((rf_params, rf_rmse))
    
    # Print out the RMSE for each parameter setting
    for params, rmse in all_rmse_results:
        print(f"Params: {params} - RMSE: {rmse:.4f}")

    # Find the best parameter combination based on RMSE
    best_rf_params, best_rf_rmse = min(all_rmse_results, key=lambda x: x[1])
    
    print(f"Best Random Forest RMSE: {best_rf_rmse:.4f} with params: {best_rf_params}")
    
    # Train the final model using the best parameters
    best_rf_model = RandomForestRegressor(random_state=42, **best_rf_params)
    best_rf_model.fit(X_train_split, y_train_split)
    
    # Save the best model
    joblib.dump(best_rf_model, 'rf_best_model_sweep.pkl')
    
    # Return the results for all parameter settings, as well as the best model and parameters
    return all_rmse_results, best_rf_model, best_rf_params, best_rf_rmse

def evaluate_and_compare_models_sweep(X_train, y_train):

    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    hist_param_grid = {
        'max_iter': [100, 500, 1000],
        'learning_rate': [0.01, 0.05, 0.1]
    }
    
    xgb_param_grid = {
        'n_estimators': [500, 1000],
        'learning_rate': [0.01, 0.05],
        'max_depth': [3, 5, 7]
    }
    from sklearn.model_selection import GridSearchCV
    # Use GridSearchCV for hyperparameter tuning
    rf_search = GridSearchCV(RandomForestRegressor(random_state=42), rf_param_grid, scoring='neg_mean_squared_error', cv=5)
    hist_search = GridSearchCV(HistGradientBoostingRegressor(random_state=42), hist_param_grid, scoring='neg_mean_squared_error', cv=5)
    xgb_search = GridSearchCV(XGBRegressor(random_state=42), xgb_param_grid, scoring='neg_mean_squared_error', cv=5)

    # Perform hyperparameter search
    rf_search.fit(X_train, y_train)
    hist_search.fit(X_train, y_train)
    xgb_search.fit(X_train, y_train)

    # Get the best model
    rf_best = rf_search.best_estimator_
    hist_best = hist_search.best_estimator_
    xgb_best = xgb_search.best_estimator_

    # Perform cross-validation predictions with the best model
    rf_pred = cross_val_predict(rf_best, X_train, y_train, cv=5)
    hist_pred = cross_val_predict(hist_best, X_train, y_train, cv=5)
    xgb_pred = cross_val_predict(xgb_best, X_train, y_train, cv=5)

    # Calculate RMSE
    rf_rmse = rmse_scorer(y_train, rf_pred)
    hist_rmse = rmse_scorer(y_train, hist_pred)
    xgb_rmse = rmse_scorer(y_train, xgb_pred)

    print(f"Random Forest Best RMSE: {rf_rmse:.4f} with params: {rf_search.best_params_}")
    print(f"HistGradientBoosting Best RMSE: {hist_rmse:.4f} with params: {hist_search.best_params_}")
    print(f"XGBRegressor Best RMSE: {xgb_rmse:.4f} with params: {xgb_search.best_params_}")

    # Save the best model
    joblib.dump(rf_best, 'rf_best_model.pkl')
    joblib.dump(hist_best, 'hist_best_model.pkl')
    joblib.dump(xgb_best, 'xgb_best_model.pkl')
    # Grid search for different weight combinations
    weight_range = np.arange(0.0, 1.0, 0.1)
    best_rmse = float('inf')
    best_weights = None
    
    for w1, w2 in product(weight_range, repeat=2):
        w3 = 1 - w1 - w2
        if w3 < 0 or w3 > 1:  # The weights must satisfy w1 + w2 + w3 = 1
            continue
        
        # Calculate combined predictions using weighted average
        combined_pred = w1 * rf_pred + w2 * hist_pred + w3 * xgb_pred
        combined_rmse = rmse_scorer(y_train, combined_pred)
        
        # Update the best combination
        if combined_rmse < best_rmse:
            best_rmse = combined_rmse
            best_weights = (w1, w2, w3)
    
    print(f"Best weights: {best_weights}")
    print(f"Combined Model RMSE with best weights: {best_rmse:.4f}")
    return rf_best, hist_best, xgb_best, best_weights

def evaluate_and_compare_models_fixed_hyper(X_train, y_train):
    # Define the best parameters
    rf_best_params = {'n_estimators': 300, 'max_depth': 20, 'min_samples_split': 2, 'min_samples_leaf': 1}
    hist_best_params = {'learning_rate': 0.05, 'max_iter': 500}
    xgb_best_params = {'n_estimators': 1000, 'learning_rate': 0.05, 'max_depth': 5}

    # Initialize the models
    rf_best = RandomForestRegressor(random_state=42, **rf_best_params)
    hist_best = HistGradientBoostingRegressor(random_state=42, **hist_best_params)
    xgb_best = XGBRegressor(random_state=42, **xgb_best_params)

    # Train the models
    rf_best.fit(X_train, y_train)
    hist_best.fit(X_train, y_train)
    xgb_best.fit(X_train, y_train)

    # Train the models and perform cross-validation predictions
    rf_pred = cross_val_predict(rf_best, X_train, y_train, cv=5)
    hist_pred = cross_val_predict(hist_best, X_train, y_train, cv=5)
    xgb_pred = cross_val_predict(xgb_best, X_train, y_train, cv=5)

    # Calculate RMSE
    rf_rmse = rmse_scorer(y_train, rf_pred)
    hist_rmse = rmse_scorer(y_train, hist_pred)
    xgb_rmse = rmse_scorer(y_train, xgb_pred)

    print(f"Random Forest Best RMSE: {rf_rmse:.4f} with params: {rf_best_params}")
    print(f"HistGradientBoosting Best RMSE: {hist_rmse:.4f} with params: {hist_best_params}")
    print(f"XGBRegressor Best RMSE: {xgb_rmse:.4f} with params: {xgb_best_params}")
    weight_range = np.arange(0.0, 1.1, 0.1)
    best_rmse = float('inf')
    best_weights = None

    for w1, w2 in product(weight_range, repeat=2):
        w3 = 1 - w1 - w2
        if w3 < 0 or w3 > 1:  # The weights must satisfy w1 + w2 + w3 = 1
            continue

        # Calculate combined predictions using weighted average
        combined_pred = w1 * rf_pred + w2 * hist_pred + w3 * xgb_pred
        combined_rmse = rmse_scorer(y_train, combined_pred)

        # Update the best combination
        if combined_rmse < best_rmse:
            best_rmse = combined_rmse
            best_weights = (w1, w2, w3)

    print(f"Best weights: {best_weights}")
    print(f"Combined Model RMSE with best weights: {best_rmse:.4f}")
    return rf_best, hist_best, xgb_best, best_weights

def make_predictions(rf_model, hist_model, xgb_model, best_weights, X_test):
    # Make predictions using the trained models
    rf_pred = rf_model.predict(X_test)
    hist_pred = hist_model.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)

    # Combine the model predictions with weighted average
    weights = best_weights
    combined_pred = weights[0] * rf_pred + weights[1] * hist_pred + weights[2] * xgb_pred

    # Create a DataFrame to store the prediction results
    ids = range(len(combined_pred))
    submission_df = pd.DataFrame({'Id': ids, 'Predicted': combined_pred})

    # Save the prediction results to a CSV file
    submission_df.to_csv('submission_v4_80_20.csv', index=False)
    print("Predictions saved to submission_combined_predictions.csv")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run EDA, preprocessing, or prediction.")
    parser.add_argument("--task", default='predict',choices=["EDA", "preprocessing", "predict"], help="Specify the task to run")
    args = parser.parse_args()

    if args.task == "EDA":
        # Load data
        train_df = pd.read_csv('raw_data/train.csv')
        test_df = pd.read_csv('raw_data/test.csv')

        # Run all EDA subtasks
        print("Running EDA...")
        print("Overview of raw training data:")
        overview(train_df)

        print("Description of raw training data:")
        description(train_df)

        print("Missing values table for training data:")
        missing_values_table(train_df, "Missing Values of Raw Training Data")

        print("Univariate analysis of training data:")
        analysis_univariate(train_df)

        print("Correlation analysis of training data:")
        analysis_correlation(train_df)

        # print("Exception detection in training data:")
        # exception_detect(train_df)

    elif args.task == "preprocessing":
        # Load data
        train_df = pd.read_csv('raw_data/train.csv')
        test_df = pd.read_csv('raw_data/test.csv')

        print("Running preprocessing...")
        generate_semantic_features(train_df)
        generate_semantic_features(test_df)
        # make_imputer(train_df)
        # make_imputer(test_df)

        cleaned_train_df = remove(train_df, ['eco_category','listing_id','title','model','description','features','accessories'])
        # cleaned_train_df = remove(train_df, ['reg_date', 'category'])
        cleaned_train_df = handle_missing_value(cleaned_train_df, threshold=50)
        cleaned_train_df = handle_mileage(cleaned_train_df)
        cleaned_train_df = handle_manufactured(cleaned_train_df)
        # cleaned_train_df = cleaned_train_df.dropna(subset=['power','road_tax'])
        cleaned_train_df.to_csv('processed_data/train.csv', index=False)
        # description(cleaned_train_df)


        cleaned_test_df = remove(test_df, ['eco_category','listing_id','title','model','description','features','accessories'])
        # cleaned_test_df = remove(test_df, ['reg_date', 'category'])
        cleaned_test_df = handle_missing_value(cleaned_test_df, threshold=50)
        cleaned_test_df = handle_mileage(cleaned_test_df)
        cleaned_test_df = handle_manufactured(cleaned_test_df)

        # description(cleaned_test_df)
        cleaned_test_df.to_csv('processed_data/test.csv', index=False)

            
    elif args.task == "predict":
        # Load processed data
        # ver = 'outlier'
        ver = 'v3'
        train_df = pd.read_csv(f'./processed_data/{ver}/train.csv')
        test_df = pd.read_csv(f'./processed_data/{ver}/test.csv')


        # Create numerical feature preprocessing function
        def numerical_preprocessing():
            return Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),  # Fill missing values with the mean
                # ('scaler', MinMaxScaler())  # Normalize the values (optional)
            ])

        # Create categorical feature preprocessing function (target encoding)
        def categorical_preprocessing():
            return Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing values with the most frequent value
                ('target_encoder', ce.TargetEncoder())  # Apply target encoding
            ])

        # Build the column transformer
        def prepare_data(train_df, test_df, numerical_features, categorical_features, target_column='price'):
            y_train = train_df[target_column]  # Extract the target column

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_preprocessing(), numerical_features),  # Preprocess numerical features
                    ('cat', categorical_preprocessing(), categorical_features)  # Preprocess categorical features
                ])
            
            # Apply the transformer to the data
            X_train = preprocessor.fit_transform(train_df, y_train)  # Pass y_train for target encoding
            X_test = preprocessor.transform(test_df)

            return X_train, X_test, y_train


        numerical_features = [    
            'manufactured', 
            # 'original_reg_date', 
            # 'reg_date', 
            # 'curb_weight', 
            'power', 
            'engine_cap', 
            'no_of_owners', 
            'depreciation', 
            'coe', 
            # 'road_tax', 
            'dereg_value', 
            'mileage', 
            'omv', 
            'arf', 
            # 'opc_scheme', 
            # 'lifespan', 
            # 'indicative_price'
            ]

        categorical_features = [
            'type_of_vehicle', 'category', 'transmission',
                                'make',
                                # 'model'
                                    # 'fuel_type'
                                    ]
        X_train, X_test, y_train = prepare_data(train_df, test_df, numerical_features, categorical_features)

        # Run prediction method
        print("Running prediction...")


        rf_model,hist_model,xgb_model, best_weights = evaluate_and_compare_models_sweep(X_train, y_train)  # 或者根据你的评估结果选择最佳模型
        
        # all_rmse_results, best_rf_model, best_rf_params, best_rf_rmse = random_forest_sweep(X_train, y_train)
        # rf_model,hist_model,xgb_model = best_rf_model,best_rf_model,best_rf_model
        # best_weights = [1,0,0]
        # rf_model = joblib.load('/storage_fast/hhbao/project/DEBUGGER/archive/v3/rf_best_model.pkl')
        # hist_model = joblib.load('/storage_fast/hhbao/project/DEBUGGER/archive/v3/hist_best_model.pkl')
        # xgb_model = joblib.load('/storage_fast/hhbao/project/DEBUGGER/archive/v3/xgb_best_model.pkl')

        make_predictions(rf_model,hist_model,xgb_model, best_weights, X_test)

        # Uncomment other methods as needed

        # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
        # random_forest(X_train, X_test, y_train, y_test)
        # logistic_regression(X_train, X_test, y_train, y_test)
        # xgboost(X_train, X_test, y_train, y_test)
        # MLP(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()