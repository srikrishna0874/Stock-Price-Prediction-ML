import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from numpy import exp
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import finnhub
import datetime as dt


def SENTIMENT_ANALYSIS(quote):

    finnhub_client = finnhub.Client(
        api_key="cnd01f1r01qr85dt36h0cnd01f1r01qr85dt36hg")

    # Load the model
    loaded_model = pickle.load(open('trained_sentiment_model.sav', 'rb'))

    # load the vectorizer
    loaded_vectorizer = pickle.load(open('tfidf_vectorizer.sav', 'rb'))
    end = dt.datetime.now().strftime("%Y-%m-%d")
    res = finnhub_client.company_news(
        'TSLA', _from="2020-06-01", to=end)
    tweets = []

    for i in res:
        tweets.append(i['headline'])

    # Preprocess the tweet strings
    preprocessed_tweets = [preprocess_tweet(tweet) for tweet in tweets]

    # Fit a new TF-IDF vectorizer to the preprocessed tweet strings
    X_new = loaded_vectorizer.transform(preprocessed_tweets)

    # Make predictions
    predictions = loaded_model.predict(X_new)

    negative_count, positive_count = 0, 0
    global_polarity = 0

    # Print predictions
    for i, prediction in enumerate(predictions):
        print(f"Tweet:{i} {tweets[i]}")
        if prediction == 0:
            negative_count += 1
            print("Sentiment: Negative")
        else:
            positive_count += 1
            print("Sentiment: Positive")
        print()

    global_polarity += positive_count
    if len(tweets) != 0:
        global_polarity /= len(tweets)

    print("Global polarity is:")
    print(global_polarity)

    if global_polarity > 0.5:
        tw_polarity = "Overall Positive"

    else:
        tw_polarity = "Overall Negative"

    labels = 'Positive', 'Negative'
    sizes = [positive_count, negative_count]
    colors = ['lightgreen', 'lightcoral']
    explode = (0, 0)

    plt.figure(figsize=(7, 7))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)
    plt.title('Sentiment Analysis Pie Chart')
    plt.axis('equal')

    plt.savefig('static/SENTIMENT_CHARTS/'+quote+'.png')

    return tweets[:5], global_polarity, tw_polarity


# Function to preprocess tweet text

def preprocess_tweet(tweet):
    port_stem = PorterStemmer()
    tweet = re.sub('[^a-zA-Z]', ' ', tweet)  # Remove non-alphabetic characters
    tweet = tweet.lower()  # Convert text to lowercase
    tweet = tweet.split()  # Tokenize text
    tweet = [port_stem.stem(word) for word in tweet if word not in stopwords.words(
        'english')]  # Remove stopwords and apply stemming
    tweet = ' '.join(tweet)
    return tweet
