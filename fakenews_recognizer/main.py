import numpy as np

from fakenews_recognizer.fake_news_data_preparation import prepare_for_mlp, prepare_for_lstm, prepare_input_for_mlp, \
    prepare_input_for_lstm, prepare_for_lstm_after_topics
from fakenews_recognizer.lstm_model import lstm_model
from fakenews_recognizer.mlp_model import mlp_model

if __name__ == '__main__':
    # train model
    x_train, y_train, x_test, y_test, features, tfidf, scaler = prepare_for_mlp()
    model = mlp_model(x_train, y_train, x_test, y_test, features)

    #x_train, y_train, x_test, y_test, amount_words, max_len, tokenizer = prepare_for_lstm()
    #model = lstm_model(x_train, y_train, x_test, y_test, amount_words, max_len)

    # predict
    while(True):
        text = input("Please enter your news article: ")
        x = prepare_input_for_mlp(text, tfidf, scaler)
        #x = prepare_input_for_lstm(text, tokenizer, max_len)
        result = model.predict(x)
        result = np.argmax(result, axis=1)
        if result == 1:
            print("fake news!")
        elif result == 0:
            print("true news!")

def predict_news(texts):

    # train model
    X_train, y_train, X_test, y_test, amount_words, max_len, tokenizer = prepare_for_lstm()
    model = lstm_model(X_train, y_train, X_test, y_test, amount_words, max_len)

    #x_train, y_train, x_test, y_test, features, tfidf, scaler = prepare_for_mlp()
    #model = mlp_model(x_train, y_train, x_test, y_test, features)

    results = []

    for text in texts:

        text = prepare_input_for_lstm(text, tokenizer, max_len)
        #text = prepare_input_for_mlp(text, tfidf, scaler)

        # predict
        result = model.predict(text)
        result = np.argmax(result, axis=1)
        results.append(result)

    return results


def fake_news_algo_with_topics(topics_model, topics, texts, predictions):
    models = []
    results = []
    
    # train model
    X_train, Y_train, X_test, Y_test, amount_words, max_len, tokenizer, len_of_categories = prepare_for_lstm_after_topics(topics_model, topics)
    index = 0
    index_after = len_of_categories[0] - 1
    for k in range(len(topics)):
        x_train = X_train[index:index_after]
        x_test = X_test[index:index_after]
        y_train = Y_train[index:index_after]
        y_test = Y_test[index:index_after]
        model = lstm_model(x_train, y_train, x_test, y_test, amount_words, max_len)
        models.append(model)
        # new indexs
        index = index_after + 1
        if k < len(topics) - 1:
            index_after = index_after + len_of_categories[k + 1]

    # predict
    for k in range(len(texts)):
        results.append(models[predictions[k]].predict(text[k]))
    return results
