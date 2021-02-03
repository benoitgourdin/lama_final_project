import re
import numpy as np
import nltk
import pandas as pd
from pandas import DataFrame
from keras.preprocessing import sequence
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.text import Tokenizer
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


def prepare_for_fnn():
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


def prepare_input_for_fnn(text, tfidf, scaler):
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


def prepare_for_lstm():
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

    # separate text and target
    all_labels = np.array(df['target'])
    all_texts = np.array(df['text'])

    # clean texts
    stop_words = set(stopwords.words('english'))
    all_cleaned_texts = np.array([clean(str(text), stop_words) for text in all_texts])

    # tokenize
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_cleaned_texts)
    all_encoded_texts = tokenizer.texts_to_sequences(all_cleaned_texts)
    all_encoded_texts = np.array(all_encoded_texts)
    amount_words = len(tokenizer.word_index) + 1

    # get the length of the longest text
    length = []
    for text in all_encoded_texts:
        length.append(len(text))
    max_len = max(length)

    # every text has the same length
    all_encoded_texts = sequence.pad_sequences(all_encoded_texts, maxlen=max_len)

    #categorize data
    all_labels = to_categorical(all_labels)

    # split data
    X_train, X_test, y_train, y_test = train_test_split(all_encoded_texts, all_labels, test_size=0.2, random_state=11)

    return X_train, y_train, X_test, y_test, amount_words, max_len, tokenizer



def prepare_input_for_lstm(text, tokenizer, max_len):
    # clean texts
    stop_words = set(stopwords.words('english'))
    cleaned_text = clean(text, stop_words)

    # tokenize
    encoded_text = tokenizer.texts_to_sequences(cleaned_text)
    encoded_text = np.array(encoded_text)

    # fit length to lstm
    encoded_text = sequence.pad_sequences(encoded_text, maxlen=max_len)

    return encoded_text


def clean(text, stop_words):
    # Removing html tags
    text = re.sub('<[^>]*>', '', text)
    # Removing emails
    text = re.sub('\S*@\S*\s?', '', text)
    # Removing urls
    text = re.sub('https?://[A-Za-z0-9]', '', text)
    # Removing numbers
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Lowering letters
    text = text.lower()
    word_tokens = word_tokenize(text)
    filtered_sentence = []
    for word_token in word_tokens:
        if word_token not in stop_words:
            filtered_sentence.append(word_token)
    # Joining words
    text = (' '.join(filtered_sentence))
    return text
