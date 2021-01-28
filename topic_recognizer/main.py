from topic_model import model
from topic_data_preparation import prepare_data, prepare_input_data


if __name__ == '__main__':
    # train model
    X_train, y_train, X_test, y_test, amount_words, max_len, tokenizer = prepare_data()
    model = model(X_train, y_train, X_test, y_test, amount_words, max_len)

    # input
    text = input("Please enter your news articles: ")
    text = prepare_input_data(text, tokenizer, max_len)
