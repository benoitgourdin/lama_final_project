from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense


def mlp_model(x_train, y_train, x_test, y_test, features):
    # mlp network
    model = Sequential()
    model.add(Dense(1000, activation='relu', input_dim=features))
    model.add(Dense(500))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(x_train, y_train, batch_size=50, epochs=10, validation_data=(x_test, y_test))
    loss, acc = model.evaluate(x_test,  y_test, verbose=2)
    print("Accuracy: " + str(acc * 100) + "%")
    return model
