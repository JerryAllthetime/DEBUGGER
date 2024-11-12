#post_EDA.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick


import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def view_outliers_after(df: pd.DataFrame, threshold: float = 1):
    """
    Plots the distribution and highlights outliers for each numeric column in the DataFrame.
    Saves each plot as an image with filenames based on the column names.
    Prints the number of outliers for each column.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - threshold (float): The multiplier for the IQR to determine outliers. Default is 1.5.
    """
    # Define the directory to save the images
    picture_folder_path = './pictures/after_processed'
    os.makedirs(picture_folder_path, exist_ok=True)  # Create directory if it does not exist

    # Get numeric columns
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    
    for col in numeric_columns:
        # Calculate IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        # Identify outliers
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)
        
        # Print outlier count
        print(f"{col}: {outlier_count} outliers detected with threshold {threshold}*IQR.")
        
        # Plot histogram for data distribution
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(df[col].dropna(), kde=True, bins=30)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')

        # Plot boxplot for outliers
        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[col])
        plt.title(f'Outliers in {col}')
        plt.xlabel(col)

        plt.tight_layout()
        
        # Save the plot as a PNG file in the specified directory
        file_path = os.path.join(picture_folder_path, f'{col}_distribution_outliers_after.png')
        plt.savefig(file_path)
        plt.close()  # Close the plot to free up memory


def overview(df: pd.DataFrame):
    # Part 1: Data overview
    print(df.info())
    print(df.head())

    # Part 2: Count the number of numerical and categorical attribute
    # numerical
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    num_numerical = len(numerical_columns)
    # categorical
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    num_categorical = len(categorical_columns)
    # Displaying the counts
    print(f"Number of numerical attributes: {num_numerical}")
    print(f"Number of categorical attributes: {num_categorical}")

    # Part 3: Calculate the number and percentage of missing values for the dataset
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / len(df)) * 100
    missing_df = pd.DataFrame({'Missing Count': missing_count, 'Percentage': missing_percentage})
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False)
    # Display the missing values statistics for the training and testing datasets
    print("Missing Values Statistics:")
    print(missing_df)

    # Part 4: Duplicate detection
    print("Duplicates in {df} Dataset：", df.duplicated().sum())

def description(df: pd.DataFrame):
    print("\n数值特征描述性统计：")
    print(df.describe())

    # 类别特征的唯一值统计
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        print(f"\n Number of unique values for class feature '{col}'")
        print(df[col].value_counts().head())

def missing_values_table(df: pd.DataFrame, title: str):
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / len(df)) * 100
    missing_df = pd.DataFrame({'Missing Count': missing_count, 'Percentage': missing_percentage})
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False)
    # Format the percentage column to 4 decimal places
    missing_df['Percentage'] = missing_df['Percentage'].apply(lambda x: f"{x:.4f}")

    fig, ax = plt.subplots(figsize=(10, len(missing_df) * 0.5 + 1))
    ax.axis('off')  # Hide axes
    ax.axis('tight')  # Make the table tight
    table = ax.table(cellText=missing_df.values, 
                     colLabels=missing_df.columns, 
                     rowLabels=missing_df.index, 
                     cellLoc='center', 
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    
    # Beautify the table with headers and cell alignment
    for key, cell in table.get_celld().items():
        cell.set_edgecolor('grey')
        if key[0] == 0 or key[1] == -1:
            cell.set_fontsize(12)
            cell.set_text_props(weight='bold')
    
    plt.title(title, fontsize=14, pad=10)
    plt.show()

def analysis_univariate(df: pd.DataFrame):
    # Plotting the price distribution with thousands separators for the x-axis
    plt.figure(figsize=(10, 6))
    sns.histplot(df['price'], bins=50, kde=True)
    plt.title('Price Distribution')
    plt.xlabel('Price')
    plt.ylabel('Frequency')

    # Adjusting the x-axis to use thousands separators
    plt.gca().xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{int(x):,}'))

    plt.show()

    # 数值特征分布
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_cols:
        plt.figure(figsize=(10, 5))
        sns.histplot(df[col], bins=30, kde=True)
        plt.title(f'{col} distribution diagram')
        plt.show()

def analysis_correlation(df: pd.DataFrame):
    # 数值特征之间的相关性
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    # 计算数值特征的相关性矩阵
    correlation_matrix = df[numeric_cols].corr()

    # 绘制热力图
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Heatmap of correlation between numerical features')
    plt.show()


    # # 类别特征与数值特征之间的关系
    # categorical_cols = df.select_dtypes(include='object').columns
    # for col in categorical_cols:
    #     if col != 'title' and col != 'description':  # 忽略文本特征
    #         plt.figure(figsize=(12, 6))
    #         sns.boxplot(x=df[col], y=df['indicative_price'])
    #         plt.title(f'{col} 与 价格之间的关系')
    #         plt.xticks(rotation=45)
    #         plt.show()

def exception_detect(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        plt.figure(figsize=(10, 5))
        sns.boxplot(df[col])
        plt.title(f'{col} boxplot - exception detect')
        plt.show()