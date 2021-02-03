import numpy as np

from topic_recognizer.lstm_model import lstm_model
from topic_recognizer.fnn_model import fnn_model
from topic_recognizer.topic_data_preparation import prepare_for_fnn, prepare_input_for_lstm, prepare_input_for_fnn, \
    prepare_for_lstm

if __name__ == '__main__':

    # define labels
    labels = ['tech', 'politics', 'sport', 'entertainment', 'business']

    # train model

    #X_train, y_train, X_test, y_test, amount_words, max_len, tokenizer = prepare_for_lstm()
    #model = lstm_model(X_train, y_train, X_test, y_test, amount_words, max_len)

    x_train, y_train, x_test, y_test, features, tfidf, scaler = prepare_for_fnn()
    model = fnn_model(x_train, y_train, x_test, y_test, features)

    while(True):
        # input
        text = input("Please enter your news articles: ")
        #text = prepare_input_for_lstm(text, tokenizer, max_len)
        text = prepare_input_for_fnn(text, tfidf, scaler)

        # predict
        result = model.predict(text)
        result = np.argmax(result, axis=1)
        print(labels[result])
