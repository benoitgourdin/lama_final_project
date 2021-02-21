from fakenews_recognizer.main import fake_news_after_topics
from topic_recognizer.main import train_topic

if __name__ == '__main__':
    topics = ['politic', 'geography', 'climate']
    texts = []
    while(start == 'yes'):
        # input
        text = input("Please enter your news article: ")
        texts.append(text)
        start = input("Some more articles ? (yes/no): ")

    # train algorithms
    topic_model = train_topic()
    predictions = []
    for text in texts:
        prediction = topic_model.predict(text)
        predictions.append(prediction)
    results = fake_news_after_topics(topic_model, topics, texts, predictions)
    for k in range(texts):
        if results[k] == 0:
            print("TRUE!")
        elif results[k] == 1:
            print("FALSE!")

