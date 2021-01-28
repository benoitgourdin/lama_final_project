import pandas as pd
import nltk
import numpy as np
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.utils.np_utils import to_categorical


def Fake_Real_News():
    # import dataset 1
    fake_df = pd.read_csv('dataset/Fake.csv')
    real_df = pd.read_csv('dataset/True.csv')
    fake_df['target'] = 1
    real_df['target'] = 0
    df = real_df
    df = df.append(fake_df, ignore_index=True)
    return df


def fake_or_real_news():
    # import dataset 1
    df_import = pd.read_csv('dataset/fake_or_real_news.csv')
    df = df_import[['text', 'label']]
    df.columns = ['text', 'target']
    for k in range(0, len(df)):
        if df['target'][k] == "FAKE":
            df['target'][k] = 1
        elif df['target'][k] == "REAL":
            df['target'][k] = 0
    return df


def data():
    # import dataset 1
    df_import = pd.read_csv('dataset/data.csv')
    df = df_import[['Body', 'Label']]
    df.columns = ['text', 'target']
    df['target'] = abs(df['target'] - 1)
    return df


def prepare_data():
    # import and short data
    # FAKE = 1, TRUE = 0
    df_1 = Fake_Real_News().sample(frac=1).reset_index(drop=True)
    df_2 = fake_or_real_news().sample(frac=1).reset_index(drop=True)
    df_3 = data().sample(frac=1).reset_index(drop=True)
    df_1 = df_1[:4000]
    df_2 = df_2[:4000]
    df_3 = df_3[:4000]
    df = df_1
    df = df.append(df_2, ignore_index=True)
    df = df.append(df_3, ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)

    # prepare text
    tokenizer = nltk.tokenize.TreebankWordTokenizer()
    stemmer = nltk.WordNetLemmatizer()
    tokenized_text = []
    for text in df['text']:
        train_tokenized = tokenizer.tokenize(str(text))
        train_lemmatized = [" ".join(stemmer.lemmatize(word) for word in train_tokenized)]
        tokenized_text.append(train_lemmatized)

    # text to numbers
    tfidf = TfidfVectorizer(stop_words='english', min_df=2, max_df=0.5, max_features=10000, ngram_range=(1, 2))
    new_df = DataFrame(tokenized_text, columns=['text'])
    vectorized_text = tfidf.fit_transform(new_df['text'])

    # scale data
    scaler = StandardScaler()
    new = vectorized_text.toarray().tolist()
    scaled_text = scaler.fit_transform(new)

    # split data
    x_train, x_test, y_train, y_test = train_test_split(scaled_text, df['target'], test_size=0.2)

    # converting data type
    y_train = np.vstack(y_train.values)
    y_test = np.vstack(y_test.values)
    features = len(tfidf.get_feature_names())
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return x_train, y_train, x_test, y_test, features, tfidf, scaler


def prepare_input_data(text, tfidf, scaler):
    # fit the input text to the data preparation
    # prepare text
    tokenizer = nltk.tokenize.TreebankWordTokenizer()
    stemmer = nltk.WordNetLemmatizer()
    tokenized_text = []
    train_tokenized = tokenizer.tokenize(str(text))
    train_lemmatized = [" ".join(stemmer.lemmatize(word) for word in train_tokenized)]
    tokenized_text.append(train_lemmatized)

    # text to numbers
    new_df = DataFrame(tokenized_text, columns=['text'])
    vectorized_text = tfidf.transform(new_df['text'])

    # scale data
    new = vectorized_text.toarray().tolist()
    scaled_text = scaler.fit_transform(new)
    return scaled_text
