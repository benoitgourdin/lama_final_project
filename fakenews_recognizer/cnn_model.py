from sklearn.metrics import accuracy_score
from tensorflow.python.keras import Sequential, regularizers
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, Dense, Embedding, GlobalMaxPooling1D
import numpy as np


def cnn_model(x_train, y_train, x_test, y_test, amount_words, max_len):
    # cnn network
    embedding_layer = Embedding(amount_words, 100, input_length=max_len, trainable=False)
    model = Sequential()
    model.add(embedding_layer)
    model.add(Conv1D(filters=64, kernel_size=4, padding='same', activation='relu'))
    model.add(MaxPooling1D())
    model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling1D())
    model.add(Conv1D(filters=256, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling1D())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(x_train, y_train, epochs=10, batch_size=60)
    predictions = model.predict(x_test)
    y_test_evaluate = np.argmax(y_test, axis=1)
    predictions = np.argmax(predictions, axis=1)
    print("Accuracy: " + str(accuracy_score(y_test_evaluate, predictions)) + "%")
    return model
