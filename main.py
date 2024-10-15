import pandas as pd
import numpy as np
import sys
from EDA import *
from preprocessing import *
from predictor import *

def main():
    if len(sys.argv) < 2:
        raise RuntimeError(
            f"Expect at least 2 arguments, got {len(sys.argv)}. "
            "Please type 'EDA', 'preprocessing' or 'predict', \n"
            "For EDA: specify a task"
            )
        return

    arg1 = sys.argv[1]

    # EDA here!
    if arg1 == 'EDA':
        # loading
        train_df = pd.read_csv('raw_data/train.csv')
        test_df = pd.read_csv('raw_data/test.csv')
        arg2 = sys.argv[2]

        # EDA code
        print("We are running EDA right now")
        if arg2 == 'overview':
            print("overview of raw training data:")
            overview(train_df)
            # print("overview of raw test data:")
            # overview(test_df)
        elif arg2 == 'description':
            print("description of raw training data:")
            description(train_df)
            # print("description of raw test data:")
            # description(test_df)
        elif arg2 == 'missing_values_table':
            missing_values_table(train_df, "Missing Values of Raw Training Data")
            # missing_values_table(test_df, "missing values of raw test data")
        elif arg2 == 'univariate_analysis':
            print("univariate analysis of training data")
            analysis_univariate(train_df)
            # print("univariate analysis of test data")
            # analysis_univariate(test_df)
        elif arg2 == 'correlation':
            print("correlation analysis of training data")
            analysis_correlation(train_df)
            # print("correlation analysis of test data")
            # analysis_correlation(test_df)
        elif arg2 == 'exception':
            print("exception detection of training data")
            exception_detect(train_df)
            # print("exception detection of test data")
            # exception_detect(test_df)
        elif arg2 == 'fillna':
            print("fill na of training data")
            fillna(train_df)
            # print("fill na of test data")
            # fillna(test_df)
        elif arg2 == 'cat_to_int':
            print("transform categorical features to integers")
        
    
    # preprocessing here!
    elif arg1 == 'preprocessing':
        # loading
        train_df = pd.read_csv('raw_data/train.csv')
        test_df = pd.read_csv('raw_data/test.csv')

        # preprocessing code
        print("We are running preprocessing right now")
        cleaned_train_df = remove(train_df, ['eco_category','listing_id','title','description','curb_weight','road_tax','features','accessories'])
        cleaned_train_df = handle_missing_value(cleaned_train_df, threshold=50)
        cleaned_train_df = handle_mileage(cleaned_train_df)
        cleaned_train_df = handle_manufactured(cleaned_train_df)
        description(cleaned_train_df)
    
    # predict here! (TODO)
    elif arg1 == 'predict':
        # should read processed data instead of raw data
        # train_df = pd.read_csv('processed_data/train.csv')
        # test_df = pd.read_csv('processed_data/test.csv')
        # X_train, X_test, y_train, y_test = predit(train_df)

        # then choose a method: logistic_regression, random_forest, xgboost, MLP...
        # logistic_regression(X_train, X_test, y_train, y_test)
        # random_forest(X_train, X_test, y_train, y_test)
        # xgboost(X_train, X_test, y_train, y_test)
        # MLP(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()