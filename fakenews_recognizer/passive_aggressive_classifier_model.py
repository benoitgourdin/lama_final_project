from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score


def train_passive_aggressive_classifier(x_train, x_test, y_train, y_test):
    # classifier
    model = PassiveAggressiveClassifier(max_iter=100)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    print("Accuracy: " + str(score * 100) + "%")

    return model
