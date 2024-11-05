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