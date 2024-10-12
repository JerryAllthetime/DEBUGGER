import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

def remove(df: pd.DataFrame, attribute_list: list):
    cleaned_df = df.drop(columns=attribute_list)
    return cleaned_df

def handle_missing_value(df: pd.DataFrame, threshold: float = 50.0):
    # 首先，对于缺失值，我们可以删除缺失值过多，且对价格没有什么影响的列，比如缺失值超过 50% 的列
    # 清洗后的数据集 重新命名为 cleaned_df 和 cleaned_df

    # Calculate the number and percentage of missing values for the training dataset
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / len(df)) * 100
    missing_df = pd.DataFrame({'Missing Count': missing_count, 'Percentage': missing_percentage})
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False)

    # Identify columns in the training dataset where the percentage of missing values is greater than the threshold
    columns_to_drop = missing_df[missing_df['Percentage'] > threshold].index

    # Drop these columns from the dataset
    cleaned_df = df.drop(columns=columns_to_drop)

    dropped_columns = columns_to_drop.tolist()
    # Using f-string format to elegantly display which columns were dropped
    dropped_columns_message = f"Columns dropped from the training dataset: {', '.join(dropped_columns)}."
    return cleaned_df

def handle_mileage(df: pd.DataFrame):
    #因为mileage时一个重要的特征，所以我们可以通过其他特征的相关性来填充缺失值

    # Plotting the distribution of 'mileage' to observe its range and detect any outliers
    plt.figure(figsize=(10, 6))
    sns.histplot(df['mileage'], bins=30, kde=True)
    plt.title('Distribution of Mileage')
    plt.xlabel('Mileage')
    plt.ylabel('Frequency')
    plt.show()

    # Checking the correlation between 'mileage' and other numerical features, especially 'manufactured' (car age)
    correlation_with_mileage = df[['mileage', 'manufactured', 'power', 'engine_cap']].corr()
    # Displaying the correlation matrix
    correlation_with_mileage
    
    # Filling missing values in 'mileage' based on the median mileage of cars grouped by 'manufactured' year
    df['mileage'] = df.groupby('manufactured')['mileage'].transform(
        lambda x: x.fillna(x.median())
    )

    # Check if there are still missing values in 'mileage', and if so, fill them with the overall median
    if df['mileage'].isnull().sum() > 0:
        overall_median_mileage = df['mileage'].median()
        df['mileage'].fillna(overall_median_mileage, inplace=True)

    # Display the count of remaining missing values in 'mileage' after imputation
    remaining_missing_mileage = df['mileage'].isnull().sum()
    remaining_missing_mileage
    return df

def handle_manufactured(df: pd.DataFrame):
    # 填补 manufactured 列的缺失值
    # 如果 manufactured 列为空，使用 registration_year - 1 填补；否则保留原值
    # Convert 'reg_date' to a datetime format and extract the year
    df['reg_date'] = pd.to_datetime(df['reg_date'], errors='coerce')
    df['registration_year'] = df['reg_date'].dt.year

    df['manufactured'] = df.apply(

        lambda row: row['registration_year'] - 1 if pd.isnull(row['manufactured']) else row['manufactured'], axis=1

    )

    # 再次检查 manufactured 列是否还有缺失值
    manufactured_missing_after = df['manufactured'].isnull().sum()

    # 查看填补后的 manufactured 列的描述性统计信息
    filled_manufactured_summary = df['manufactured'].describe()
    manufactured_missing_after
    return df