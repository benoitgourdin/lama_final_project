import numpy as np

from fakenews_recognizer.fake_news_data_preparation import prepare_for_fnn, prepare_for_lstm, prepare_input_for_fnn
from fakenews_recognizer.lstm_model import lstm_model
from fakenews_recognizer.fnn_model import fnn_model
from fakenews_recognizer.passive_aggressive_model import train_passive_aggressive_classifier

if __name__ == '__main__':
    # train model
    x_train, y_train, x_test, y_test, features, tfidf, scaler = prepare_for_fnn()
    model = fnn_model(x_train, y_train, x_test, y_test, features)
    #model = train_passive_aggressive_classifier(x_train, x_test, y_train, y_test)

    #X_train, y_train, X_test, y_test, amount_words, max_len, tokenizer = prepare_for_lstm()
    #model = lstm_model(X_train, y_train, X_test, y_test, amount_words, max_len)

    # predict
    while(True):
        text = input("Please enter your news article: ")
        x = prepare_input_for_fnn(text, tfidf, scaler)
        #x = prepare_input_for_lstm(text, tokenizer, max_len)
        result = model.predict(x)
        result = np.argmax(result, axis=1)
        if result == 1:
            print("fake news!")
        elif result == 0:
            print("true news!")
