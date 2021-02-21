from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Embedding, LSTM


def lstm_model(x_train, y_train, x_test, y_test, amount_words, max_len):
    # lstm network
    model = Sequential()
    model.add(Embedding(input_dim=amount_words, output_dim=50, input_length=max_len))
    model.add(LSTM(50, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(x_train, y_train, epochs=10, batch_size=50)
    loss, acc = model.evaluate(x_test,  y_test, verbose=2)
    print("Accuracy: " + str(acc * 100) + "%")
    return model
