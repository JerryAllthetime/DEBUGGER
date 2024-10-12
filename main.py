import pandas as pd
import numpy as np
import sys
from EDA import overview, description, missing_values_table, exception_detect, analysis_univariate, analysis_correlation
from preprocessing import remove, handle_missing_value, handle_mileage, handle_manufactured



def main():
    if len(sys.argv) < 2:
        raise RuntimeError(
            f"Expect at least 2 arguments, got {len(sys.argv)}. "
            "Please type 'EDA' or 'preprocessing', \n"
            "For EDA: specify a task from 'overview', 'description', 'missing_values_table', 'univariate_analysis', 'correlation', 'exception'"
            )
        return
    
    # loading
    train_df = pd.read_csv('raw_data/train.csv')
    test_df = pd.read_csv('raw_data/test.csv')

    arg1 = sys.argv[1]

    # EDA here!
    if arg1 == 'EDA':
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
    
    # preprocessing here!
    elif arg1 == 'preprocessing':
        # preprocessing code
        print("We are running preprocessing right now")
        cleaned_train_df = remove(train_df, ['eco_category','listing_id','title','description','curb_weight','road_tax','features','accessories'])
        cleaned_train_df = handle_missing_value(cleaned_train_df, threshold=50)
        cleaned_train_df = handle_mileage(cleaned_train_df)
        cleaned_train_df = handle_manufactured(cleaned_train_df)
        description(cleaned_train_df)
    
    # predict here! (TODO)

if __name__ == "__main__":
    main()