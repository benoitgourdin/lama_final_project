import pandas as pd
import numpy as np
from pandas import DataFrame

import fakenews_recognizer as fn
from sklearn.model_selection import train_test_split

from topic_recognizer.mlp_model import mlp_model
from topic_recognizer.topic_data_preparation import prepare_for_mlp, prepare_input_for_mlp


def Fake_Real_News():
    # import dataset 1
    fake_df = pd.read_csv('dataset/Fake.csv')
    real_df = pd.read_csv('dataset/True.csv')
    fake_df['target'] = 1
    real_df['target'] = 0
    df = real_df
    df = df.append(fake_df, ignore_index=True)
    return df


def fake_or_real_news():
    # import dataset 1
    df_import = pd.read_csv('dataset/fake_or_real_news.csv')
    df = df_import[['text', 'label']]
    df.columns = ['text', 'target']
    for k in range(0, len(df)):
        if df['target'][k] == "FAKE":
            df['target'][k] = 1
        elif df['target'][k] == "REAL":
            df['target'][k] = 0
    return df


def data():
    # import dataset 1
    df_import = pd.read_csv('dataset/data.csv')
    df = df_import[['Body', 'Label']]
    df.columns = ['text', 'target']
    df['target'] = abs(df['target'] - 1)
    return df


if __name__ == '__main__':

    # train topic recognizer
    x_train, y_train, x_test, y_test, features, tfidf, scaler = prepare_for_mlp()
    model = mlp_model(x_train, y_train, x_test, y_test, features)

    # import fake news datset
    df_1 = Fake_Real_News().sample(frac=1).reset_index(drop=True)
    df_2 = fake_or_real_news().sample(frac=1).reset_index(drop=True)
    df_3 = data().sample(frac=1).reset_index(drop=True)
    df_1 = df_1[:4000]
    df_2 = df_2[:4000]
    df_3 = df_3[:4000]
    df = df_1
    df = df.append(df_2, ignore_index=True)
    df = df.append(df_3, ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)

    # create dataframes for topics
    df0 = DataFrame(columns=['text', 'target'])
    df1 = DataFrame(columns=['text', 'target'])
    df2 = DataFrame(columns=['text', 'target'])
    df3 = DataFrame(columns=['text', 'target'])
    df4 = DataFrame(columns=['text', 'target'])

    # predict topic of dataset
    for i in range(len(df)):
        text = df['text'][i]
        text = prepare_input_for_mlp(text, tfidf, scaler)
        result = model.predict(text)
        result = np.argmax(result, axis=1)
        if result == 0:
            df0.append(df.iloc[[i]])
        if result == 1:
            df1.append(df.iloc[[i]])
        if result == 2:
            df2.append(df.iloc[[i]])
        if result == 3:
            df3.append(df.iloc[[i]])
        if result == 4:
            df4.append(df.iloc[[i]])

    # train a model for each topic
    x_train_0, x_test_0, y_train_0, y_test_0 = train_test_split(df0['text'], df0['target'], test_size=0.2)
    x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(df1['text'], df1['target'], test_size=0.2)
    x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(df2['text'], df2['target'], test_size=0.2)
    x_train_3, x_test_3, y_train_3, y_test_3 = train_test_split(df3['text'], df3['target'], test_size=0.2)
    x_train_4, x_test_4, y_train_4, y_test_4 = train_test_split(df4['text'], df4['target'], test_size=0.2)

    model0 = fn.mlp_model.mlp_model(x_train_0, x_test_0, y_train_0, y_test_0, features)
    model1 = fn.mlp_model.mlp_model(x_train_1, x_test_1, y_train_1, y_test_1, features)
    model2 = fn.mlp_model.mlp_model(x_train_2, x_test_2, y_train_2, y_test_2, features)
    model3 = fn.mlp_model.mlp_model(x_train_3, x_test_3, y_train_3, y_test_3, features)
    model4 = fn.mlp_model.mlp_model(x_train_4, x_test_4, y_train_4, y_test_4, features)

    # predict input
    text = input("Please enter your news articles: ")
    text = prepare_input_for_mlp(text, tfidf, scaler)
    result = model.predict(text)
    result = np.argmax(result, axis=1)
    if result == 0:
        result2 = model0.predict(text)
    if result == 1:
        result2 = model1.predict(text)
    if result == 2:
        result2 = model2.predict(text)
    if result == 3:
        result2 = model3.predict(text)
    if result == 4:
        result2 = model4.predict(text)
    result2 = np.argmax(result2, axis=1)
    if result2 == 1:
        print("fake news!")
    elif result2 == 0:
        print("true news!")
