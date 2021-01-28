import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
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


def prepare_data():
    # import data
    df = pd.read_csv('dataset/bbc-text.csv')

    # define labels
    labels = ['tech', 'politics', 'sport', 'entertainment', 'business']

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


def prepare_input_data(text, tokenizer, max_len):
    # clean texts
    stop_words = set(stopwords.words('english'))
    cleaned_text = clean(text, stop_words)

    # tokenize
    encoded_text = tokenizer.texts_to_sequences(cleaned_text)
    encoded_text = np.array(encoded_text)

    # fit length to lstm
    encoded_text = sequence.pad_sequences(encoded_text, maxlen=max_len)

    return encoded_text
