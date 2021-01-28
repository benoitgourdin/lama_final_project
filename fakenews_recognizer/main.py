from fake_news_data_preparation import prepare_data, prepare_input_data
from fake_news_model import model
from passive_aggressive_classifier_model import train_passive_aggressive_classifier

if __name__ == '__main__':
    # train model
    x_train, y_train, x_test, y_test, features, tfidf, scaler = prepare_data()
    model = model(x_train, y_train, x_test, y_test, features)
    #model = train_passive_aggressive_classifier(x_train, x_test, y_train, y_test)

    # predict
    text = input("Please enter your news article: ")
    x = prepare_input_data(text, tfidf, scaler)
    result = model.predict(x)
    if result == 1:
        print("fake news!")
    elif result == 0:
        print("true news!")
