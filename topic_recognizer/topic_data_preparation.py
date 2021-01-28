import re
import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.utils.np_utils import to_categorical


def clean(text, stop_words):
    # Lowering letters
    text = text.lower()
    # Removing html tags
    text = re.sub('<[^>]*>', '', text)
    # Removing emails
    text = re.sub('\S*@\S*\s?', '', text)
    # Removing urls
    text = re.sub('https?://[A-Za-z0-9]', '', text)
    # Removing numbers
    text = re.sub('[^a-zA-Z]', ' ', text)
    word_tokens = word_tokenize(text)
    filtered_sentence = []
    for word_token in word_tokens:
        if word_token not in stop_words:
            filtered_sentence.append(word_token)

    # Joining words
    text = (' '.join(filtered_sentence))
    return text


def prepare_for_lstm():
    # import data
    df = pd.read_csv('dataset/bbc-text.csv')

    # separate text and target
    all_labels = np.array(df['category'])
    all_texts = np.array(df['text'])

    # clean texts
    stop_words = set(stopwords.words('english'))
    all_cleaned_texts = np.array([clean(text, stop_words) for text in all_texts])

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

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_labels)
    all_encoded_labels = tokenizer.texts_to_sequences(all_labels)
    all_encoded_labels = np.array(all_encoded_labels)

    all_labels = to_categorical(all_encoded_labels)[:, 1:]

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


def prepare_for_mlp():
    # import data
    df = pd.read_csv('dataset/bbc-text.csv')

    ###tech = 0, politics = 1, sport = 2, entertainment = 3, business = 4

    tokenizer = nltk.tokenize.TreebankWordTokenizer()
    stemmer = nltk.WordNetLemmatizer()

    # prepare text
    tokenized_text = []

    for text in df['text']:
        train_tokenized = tokenizer.tokenize(str(text))
        train_lemmatized = [" ".join(stemmer.lemmatize(word) for word in train_tokenized)]
        tokenized_text.append(train_lemmatized)

    # text to numbers
    tfidf = TfidfVectorizer(stop_words='english', min_df=2, max_df=0.5, max_features=10000, ngram_range=(1, 2))
    new_df = DataFrame(tokenized_text, columns=['text'])
    vectorized_text = tfidf.fit_transform(new_df['text'])

    scaler = StandardScaler()
    new = vectorized_text.toarray().tolist()
    scaled_text = scaler.fit_transform(new)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['category'])
    all_encoded_labels = tokenizer.texts_to_sequences(df['category'])
    all_encoded_labels = np.array(all_encoded_labels)

    x_train, x_test, y_train, y_test = train_test_split(scaled_text, all_encoded_labels, test_size=0.2)

    features = len(tfidf.get_feature_names())

    y_train = to_categorical(y_train)[:, 1:]
    y_test = to_categorical(y_test)[:, 1:]

    return x_train, y_train, x_test, y_test, features


def prepare_input_for_mlp(text, tfidf, scaler):
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

