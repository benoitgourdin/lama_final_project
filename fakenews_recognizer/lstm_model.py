from sklearn.metrics import accuracy_score
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Embedding, LSTM
import numpy as np


def lstm_model(x_train, y_train, x_test, y_test, amount_words, max_len):
    # ann network
    model = Sequential()
    model.add(Embedding(input_dim=amount_words, output_dim=128, input_length=max_len))
    model.add(LSTM(100))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(x_train, y_train, epochs=10, batch_size=60)
    predictions = model.predict(x_test)
    y_test_evaluate = np.argmax(y_test, axis=1)
    predictions = np.argmax(predictions, axis=1)
    print("Accuracy: " + str(accuracy_score(y_test_evaluate, predictions)) + "%")
    return model
