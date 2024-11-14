## Project Overview

This repository contains code that serves as an extension and enhancement to the main branch, focusing on data preprocessing and data analysis. The code includes advanced handling of missing values in the preprocessing module and provides additional data analysis on the processed data, aiming to improve data quality for subsequent model training.

## Functionality

This code introduces the following improvements in data handling and analysis:

### Data Preprocessing

- **Missing Value Imputation**: Comprehensive handling of missing values with various imputation techniques, including:
  - **K-Nearest Neighbors (KNN) Imputation**: For certain numerical missing values, ensuring preservation of multidimensional feature relationships.
  - **Mapping-Based Imputation**: Logical mapping based on relationships between columns (e.g., calculating the manufacturing year based on the registration year).
  - **Grouped Median/Mean Imputation**: Filling missing values based on grouped median or mean values, helping to maintain distribution characteristics.
  - **Mode Imputation**: Used for categorical attributes to preserve original distribution.

- **Outlier Handling**: Identifies and manages outliers based on the Interquartile Range (IQR) method for numerical features.

### Data Analysis

- Provides further insights into the processed dataset, including:
  - **Data Overview**: Statistical information on numerical and categorical features.
  - **Univariate Analysis, Correlation Analysis, and Descriptive Statistics**: To explore feature distributions and relationships.
  - **Outlier Detection**: Assists in identifying and managing outliers in the dataset.
