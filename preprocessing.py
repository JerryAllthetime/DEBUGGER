#preprocessing.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
from sklearn.preprocessing import LabelEncoder
import os
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from pandas import DataFrame
import nltk
import pandas as pd
import numpy as np
import string
import pickle
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import string
import pickle
import nltk
from nltk import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import pandas as pd
import nltk
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.decomposition import PCA

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

def remove(df: pd.DataFrame, attribute_list: list):
    cleaned_df = df.drop(columns=attribute_list)
    return cleaned_df

def fillna(df: DataFrame):
    df['type_of_vehicle'].fillna('Unknown', inplace=True)
    df['category'].fillna('Unknown', inplace=True)
    df['transmission'].fillna('Unknown', inplace=True)
    df['eco_category'].fillna('Unknown', inplace=True)
    df['make'].fillna('Unknown', inplace=True)
    return df

def cat_to_int(df: DataFrame, ):
    # Initialize label encoders
    le_type_of_vehicle = LabelEncoder()
    le_category = LabelEncoder()
    le_transmission = LabelEncoder()
    le_eco_category = LabelEncoder()
    le_make = LabelEncoder()

    # Fit and transform the categories to integers
    df['type_of_vehicle'] = le_type_of_vehicle.fit_transform(df['type_of_vehicle'])
    df['category'] = le_category.fit_transform(df['category'])
    df['transmission'] = le_transmission.fit_transform(df['transmission'])
    df['eco_category'] = le_eco_category.fit_transform(df['eco_category'])
    df['make'] = le_make.fit_transform(df['make'])
    return df


def preprocess_text(s, lowercase=True, remove_stopwords=True, remove_punctuation=True, stemmer=None, lemmatizer=None):
        # TODO
    # nltk.download('punkt', download_dir=download_dir)
    # nltk.download('punkt_tab', download_dir=download_dir)
    # nltk.download('wordnet', download_dir=download_dir)
    # nltk.download('stopwords', download_dir=download_dir)
    # nltk.download('omw-1.4', download_dir=download_dir)
    # nltk.download('averaged_perceptron_tagger', download_dir=download_dir)
    nltk.download('punkt')
    nltk.download('punkt_tab')

    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('omw-1.4')
    nltk.download('averaged_perceptron_tagger')

    nltk_stopwords = set(stopwords.words('english'))
    nltk_stopwords.remove('no')
    nltk_stopwords.remove('not')

    porter_stemmer = PorterStemmer()
    wordnet_lemmatizer = WordNetLemmatizer()
    punctuation_translator = str.maketrans('', '', string.punctuation)
    tokens = word_tokenize(s)

    if lemmatizer is not None:
        tokens = lemmatize_tokens(lemmatizer, tokens)
    elif stemmer is not None:
        tokens = stem_tokens(stemmer, tokens)

    if lowercase:
        tokens = [token.lower() for token in tokens]

    if remove_stopwords:
        tokens = [token for token in tokens if not token in nltk_stopwords]

    # Remove all punctuation marks if needed (note: also converts, e.g, "Mr." to "Mr")
    if remove_punctuation:
        tokens = [ ''.join(c for c in s if c not in string.punctuation) for s in tokens ]
        tokens = [ token for token in tokens if len(token) > 0 ] # Remove "empty" tokens

    return ' '.join(tokens)

# def remove_punctuation(s):
#     return s.translate(punctuation_translator)

def lemmatize_tokens(lemmatizer, tokens):
    pos_tag_list = pos_tag(tokens)
    for idx, (token, tag) in enumerate(pos_tag_list):
        tag_simple = tag[0].lower() # Converts, e.g., "VBD" to "c"
        if tag_simple in ['n', 'v', 'j']:
            word_type = tag_simple.replace('j', 'a')
        else:
            word_type = 'n'
        lemmatized_token = lemmatizer.lemmatize(token, pos=word_type)
        tokens[idx] = lemmatized_token
    return tokens

def stem_tokens(stemmer, tokens):
    for idx, token in enumerate(tokens):
        tokens[idx] = stemmer.stem(token)
    return tokens

def preprocess_numerical(df: DataFrame, feature):
    df[feature] = df[feature].fillna(df[feature].median())
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[feature] = scaler.fit_transform(df[[feature]])
    return df

def handle_missing_value(df: pd.DataFrame, threshold: float = 50.0):

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
    #Because mileage is an important feature, we can fill in missing values by the correlation between other features.

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
    # Fill missing values in the manufactured column
    # If the manufactured column is empty, fill it with registration_year - 1; otherwise, leave the original value unchanged.
    # Convert 'reg_date' to a datetime format and extract the year
    df['reg_date'] = pd.to_datetime(df['reg_date'], errors='coerce')
    df['registration_year'] = df['reg_date'].dt.year

    df['manufactured'] = df.apply(

        lambda row: row['registration_year'] - 1 if pd.isnull(row['manufactured']) else row['manufactured'], axis=1

    )

    # Double check the manufactured column for missing values
    manufactured_missing_after = df['manufactured'].isnull().sum()

    # View descriptive statistics for the filled manufactured columns
    filled_manufactured_summary = df['manufactured'].describe()
    manufactured_missing_after
    return df


def generate_semantic_features(df: pd.DataFrame):
    # open the pickle file
    with open('./semantic_file/car_price.pkl', 'rb') as f:
        car_price = pickle.load(f)

    with open('./semantic_file/car_embedding.pkl', 'rb') as f:
        car_embedding = pickle.load(f)
        
    df['estimated_price'] = df['title'].map(car_price)
    df['title_embedding'] = df['title'].map(car_embedding)

    # Assuming 'df' is your DataFrame and 'title_embedding' is a list of embeddings in each row
    # Extract embeddings and apply PCA
    embeddings = list(data['title_embedding'])
    pca = PCA(n_components=10)
    reduced_embeddings = pca.fit_transform(embeddings)

    # Create DataFrame for reduced embeddings and join with original DataFrame
    reduced_df = pd.DataFrame(reduced_embeddings, columns=[f'embedding_dim_{i+1}' for i in range(10)])
    data = data.join(reduced_df)

    # Drop the original 'title_embedding' column if no longer needed
    data = data.drop(columns=['title_embedding'])
    return df


def remove_outliers(df: pd.DataFrame, threshold: float = 1.5) -> pd.DataFrame:
    """
    Removes outliers from each numeric column in the DataFrame based on the IQR method.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - threshold (float): The multiplier for the IQR to determine outliers. Default is 1.5.
    
    Returns:
    - pd.DataFrame: The DataFrame with outliers removed.
    """
    # Make a copy of the DataFrame to avoid modifying the original data
    df_cleaned = df.copy()
    
    # Iterate through each numeric column
    numeric_columns = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_columns:
        print(f"Removing outliers with threshold {threshold}*IQR for column {col}")
        # Calculate IQR for the column
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        # Filter out the rows containing outliers
        before_removal = df_cleaned.shape[0]
        df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
        after_removal = df_cleaned.shape[0]
        
        # Print the number of outliers removed for this column
        print(f"{col}: Removed {before_removal - after_removal} outliers with threshold {threshold}*IQR.")
    
    return df_cleaned

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def view_outliers_before(df: pd.DataFrame, threshold: float = 1.5):
    """
    Plots the distribution and highlights outliers for each numeric column in the DataFrame.
    Saves each plot as an image with filenames based on the column names.
    Prints the number of outliers for each column.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - threshold (float): The multiplier for the IQR to determine outliers. Default is 1.5.
    """
    # Define the directory to save the images
    picture_folder_path = './pictures/before_processed'
    os.makedirs(picture_folder_path, exist_ok=True)  # Create directory if it does not exist
    print(f"Directory '{picture_folder_path}' created successfully.")

    # Get numeric columns
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    
    for col in numeric_columns:
        # Calculate IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")
        
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
        file_path = os.path.join(picture_folder_path, f'{col}_distribution_outliers_before.png')
        plt.savefig(file_path)
        plt.close()  # Close the plot to free up memory

# Fill the make column according to the map of model -> make
def handlemiss_make(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing values in the 'make' column by looking up the 'model' in an internally generated dictionary.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'make' and 'model' columns.

    Returns:
    - pd.DataFrame: DataFrame with missing 'make' values filled where possible.
    """
    # Create a model-to-make dictionary directly within the function
    model_to_make = {model: make for make, models in df.groupby('make')['model'] for model in models.unique()}

    # Fill missing 'make' values by looking up the model in the generated dictionary
    for index, row in df.iterrows():
        if pd.isnull(row['make']) and pd.notnull(row['model']):
            model = row['model']
            if model in model_to_make:
                df.at[index, 'make'] = model_to_make[model]

    print("Missing 'make' values have been filled where possible based on 'model'")
    return df



def fill_power_with_knn(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses KNN to impute missing values in the 'power' column without pre-filtering rows with missing values 
    in 'engine_cap', 'depreciation', and 'dereg_value'.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'power', 'engine_cap', 'depreciation', 'dereg_value', 
                         'make', 'model', and 'type_of_vehicle' columns.

    Returns:
    - pd.DataFrame: DataFrame with missing 'power' values filled.
    """
    # Select the feature column for padding
    columns_to_use = ['power', 'engine_cap', 'depreciation', 'dereg_value', 'make', 'model', 'type_of_vehicle']
    df_subset = df[columns_to_use].copy()

    # Label the classification features
    label_encoders = {}
    for col in ['make', 'model', 'type_of_vehicle']:
        le = LabelEncoder()
        df_subset[col] = le.fit_transform(df_subset[col].astype(str).fillna("Unknown"))  
        label_encoders[col] = le

    # Standardized numerical features
    scaler = StandardScaler()
    numeric_features = ['power', 'engine_cap', 'depreciation', 'dereg_value']
    df_subset[numeric_features] = scaler.fit_transform(df_subset[numeric_features])

    # Fill in missing values with KNNImputer
    knn_imputer = KNNImputer(n_neighbors=5)
    df_imputed = knn_imputer.fit_transform(df_subset)

    # Restore the filled data
    df_imputed = pd.DataFrame(df_imputed, columns=columns_to_use)
    df_imputed[numeric_features] = scaler.inverse_transform(df_imputed[numeric_features])

    # Assigns the populated 'power' column back to the original dataset
    df['power'] = df_imputed['power']

    print("Missing 'power' values have been filled using KNN imputation.")
    return df


def fill_engine_cap_with_knn(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses KNN to impute missing values in the 'engine_cap' column based on 'make', 'model', 
    'type_of_vehicle', 'power', and 'price' columns.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'engine_cap', 'make', 'model', 'type_of_vehicle', 
                         'power', and 'price' columns.

    Returns:
    - pd.DataFrame: DataFrame with missing 'engine_cap' values filled.
    """
    # Select the feature column for padding
    columns_to_use = ['engine_cap', 'make', 'model', 'type_of_vehicle', 'power', 'price']
    df_subset = df[columns_to_use].copy()

    # Label the classification features
    label_encoders = {}
    for col in ['make', 'model', 'type_of_vehicle']:
        le = LabelEncoder()
        df_subset[col] = le.fit_transform(df_subset[col].astype(str).fillna("Unknown")) 
        label_encoders[col] = le

    # Standardized numerical features
    scaler = StandardScaler()
    numeric_features = ['engine_cap', 'power', 'price']
    df_subset[numeric_features] = scaler.fit_transform(df_subset[numeric_features])

    # Fill in missing values with KNNImputer
    knn_imputer = KNNImputer(n_neighbors=5)
    df_imputed = knn_imputer.fit_transform(df_subset)

    # Restore the filled data
    df_imputed = pd.DataFrame(df_imputed, columns=columns_to_use)
    df_imputed[numeric_features] = scaler.inverse_transform(df_imputed[numeric_features])

    # Assign the populated 'engine_cap' column back to the original data set
    df['engine_cap'] = df_imputed['engine_cap']

    print("Missing 'engine_cap' values have been filled using KNN imputation.")
    return df

# df = fill_engine_cap_with_knn(df)

def fill_depreciation_with_knn(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses KNN to impute missing values in the 'depreciation' column based on relevant columns.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'depreciation' and other related columns.

    Returns:
    - pd.DataFrame: DataFrame with missing 'depreciation' values filled.
    """
    # Select the feature column for padding
    columns_to_use = ['depreciation', 'make', 'model', 'manufactured', 'type_of_vehicle', 'power', 'engine_cap', 'price']
    df_subset = df[columns_to_use].copy()

    # Label the classification features
    label_encoders = {}
    for col in ['make', 'model', 'type_of_vehicle']:
        le = LabelEncoder()
        df_subset[col] = le.fit_transform(df_subset[col].astype(str).fillna("Unknown"))  
        label_encoders[col] = le

    # Standardized numerical features
    scaler = StandardScaler()
    numeric_features = ['depreciation', 'manufactured', 'power', 'engine_cap', 'price']
    df_subset[numeric_features] = scaler.fit_transform(df_subset[numeric_features])

    # Fill in missing values with KNNImputer
    knn_imputer = KNNImputer(n_neighbors=5)
    df_imputed = knn_imputer.fit_transform(df_subset)

    # Restore the filled data
    df_imputed = pd.DataFrame(df_imputed, columns=columns_to_use)
    df_imputed[numeric_features] = scaler.inverse_transform(df_imputed[numeric_features])


    df['depreciation'] = df_imputed['depreciation']

    print("Missing 'depreciation' values have been filled using KNN imputation.")
    return df

def fill_omv_with_knn(df: pd.DataFrame) -> pd.DataFrame:
    # Select features for imputation, including the target column 'omv'
    features = ['make', 'model', 'type_of_vehicle', 'power', 'price', 'omv']
    df_knn = df[features].copy()

    # Encode categorical variables
    label_encoders = {}
    for col in ['make', 'model', 'type_of_vehicle']:
        le = LabelEncoder()
        df_knn[col] = le.fit_transform(df_knn[col].astype(str))  # Convert to string to avoid NaN values
        label_encoders[col] = le

    # Initialize KNNImputer
    knn_imputer = KNNImputer(n_neighbors=5)

    # Use KNNImputer to fill 'omv' column
    df_knn_imputed = knn_imputer.fit_transform(df_knn)

    # Assign the filled data back to the original DataFrame
    df['omv'] = df_knn_imputed[:, features.index('omv')]

    return df

def fill_arf_with_knn(df: pd.DataFrame) -> pd.DataFrame:
    # Select features for imputation, including the target column 'arf'
    features = ['make', 'model', 'type_of_vehicle', 'power', 'price', 'omv', 'arf']
    df_knn = df[features].copy()

    # Encode categorical variables
    label_encoders = {}
    for col in ['make', 'model', 'type_of_vehicle']:
        le = LabelEncoder()
        df_knn[col] = le.fit_transform(df_knn[col].astype(str))  # Convert to string to avoid NaN values
        label_encoders[col] = le

    # Initialize KNNImputer
    knn_imputer = KNNImputer(n_neighbors=5)

    # Use KNNImputer to fill 'arf' column
    df_knn_imputed = knn_imputer.fit_transform(df_knn)

    # Assign the filled data back to the original DataFrame
    df['arf'] = df_knn_imputed[:, features.index('arf')]

    return df

def fill_dereg_value_with_knn(df: pd.DataFrame) -> pd.DataFrame:
    # Select features for imputation, including the target column 'dereg_value'
    features = ['make', 'model', 'type_of_vehicle', 'power', 'price', 'omv', 'arf', 'dereg_value']
    df_knn = df[features].copy()

    # Encode categorical variables
    label_encoders = {}
    for col in ['make', 'model', 'type_of_vehicle']:
        le = LabelEncoder()
        df_knn[col] = le.fit_transform(df_knn[col].astype(str))  # Convert to string to avoid NaN values
        label_encoders[col] = le

    # Initialize KNNImputer
    knn_imputer = KNNImputer(n_neighbors=5)

    # Use KNNImputer to fill 'dereg_value' column
    df_knn_imputed = knn_imputer.fit_transform(df_knn)

    # Assign the filled data back to the original DataFrame
    df['dereg_value'] = df_knn_imputed[:, features.index('dereg_value')]

    return df

def fill_no_of_owners_with_mode(df: pd.DataFrame) -> pd.DataFrame:
    # Calculate the mode of 'no_of_owners'
    mode_value = df['no_of_owners'].mode()[0]
    
    # Fill missing values with the mode
    df['no_of_owners'].fillna(mode_value, inplace=True)
    
    return df

# Usage example
# df = fill_no_of_owners_with_mode(df)

