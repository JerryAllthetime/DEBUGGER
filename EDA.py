import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

def overview(df: pd.DataFrame):
    # Part 1: Data overview
    print(df.info())
    print(df.head())

    # Part 2: Count the number of numerical and categorical attributes
    # Numerical
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    num_numerical = len(numerical_columns)
    # Categorical
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
    print(f"Duplicates in Dataset: {df.duplicated().sum()}")

def description(df: pd.DataFrame):
    print("\nDescriptive statistics for numerical features:")
    print(df.describe())

    # Unique values for categorical features
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        print(f"\n Number of unique values for categorical feature '{col}':")
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

    # Distribution of numerical features
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_cols:
        plt.figure(figsize=(10, 5))
        sns.histplot(df[col], bins=30, kde=True)
        plt.title(f'{col} Distribution')
        plt.show()

def analysis_correlation(df: pd.DataFrame):
    # Correlation between numerical features
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    # Calculate the correlation matrix for numerical features
    correlation_matrix = df[numeric_cols].corr()

    # Plotting the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Heatmap of Correlation between Numerical Features')
    plt.show()

    # Relationship between categorical and numerical features
    # categorical_cols = df.select_dtypes(include='object').columns
    # for col in categorical_cols:
    #     if col != 'title' and col != 'description':  # Ignore text features
    #         plt.figure(figsize=(12, 6))
    #         sns.boxplot(x=df[col], y=df['indicative_price'])
    #         plt.title(f'Relationship between {col} and Price')
    #         plt.xticks(rotation=45)
    #         plt.show()

def exception_detect(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        plt.figure(figsize=(10, 5))
        sns.boxplot(df[col])
        plt.title(f'{col} Boxplot - Outlier Detection')
        plt.show()
