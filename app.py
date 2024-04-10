import re
import math
import requests
import yfinance as yf
from flask import Flask, Response, render_template, redirect, send_file, url_for, request, send_from_directory
import datetime as dt
import pandas as pd
from ML_models.lstm import *
from ML_models.arima import *
from ML_models.linear_regression import *
from ML_models.sentiment import *
from Firebase.firebase import *
import json
from datetime import date
import finnhub
from nltk.stem.porter import PorterStemmer
from numpy import exp
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up',
             'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]


def final_result(global_polarity, today_stock, mean):
    if today_stock.iloc[-1]['Close'] < mean:
        if global_polarity > 0.5:
            idea = "RISE"
            decision = "BUY"
            print()
            print(
                "##############################################################################")
            print("According to the ML Predictions and Sentiment Analysis of Tweets, a",
                  idea, "is expected => ", decision)
        elif global_polarity <= 0.5:
            idea = "FALL"
            decision = "SELL"
            print()
            print(
                "##############################################################################")
            print("According to the ML Predictions and Sentiment Analysis of Tweets, a",
                  idea, "is expected => ", decision)
    else:
        idea = "FALL"
        decision = "SELL"
        print()
        print(
            "##############################################################################")
        print("According to the ML Predictions and Sentiment Analysis of Tweets, a",
              idea, "is expected => ", decision)
    return idea, decision


def fetching_required_dataset(quote):
    end = dt.datetime.now()
    start = dt.datetime(end.year-2, end.month, end.day)
    data = yf.download(quote, start=start, end=end)
    df = pd.DataFrame(data=data)
    df.to_csv(
        'Datasets/'+str(dt.datetime.now().strftime("%Y-%m-%d"))+quote+'.csv')
    df = pd.read_csv(
        'Datasets/'+str(dt.datetime.now().strftime("%Y-%m-%d"))+quote+'.csv')

    print(
        "##############################################################################")
    print("Today's", quote, "Stock Data: ")
    today_stock = data.iloc[-1:]
    print(today_stock)
    print(
        "##############################################################################")
    df = df.dropna()
    code_list = []
    for i in range(0, len(df)):
        code_list.append(quote)
    df2 = pd.DataFrame(code_list, columns=['Code'])
    df2 = pd.concat([df2, df], axis=1)
    df = df2
    return df, today_stock


def common_ml_code(quote):

    df, today_stock = fetching_required_dataset(quote)
    # type: ignore
    error_arima, arima_pred = ARIMA_ALGORITHM(df, quote)  # type: ignore
    linear_regression_pred, error_linear_regression, forecast_set = LINEAR_REGRESSION_ALGORITHM(
        df, quote)
    # recent_tweets, global_polarity, tw_polarity = SENTIMENT_ANALYSIS(quote)
    lstm_pred, error_lstm, recent_tweets, global_polarity, tw_polarity = LSTM_ALGORITHM(
        df, quote)
    mean = forecast_set.mean()
    list_of_predictions_using_lr = []
    for i in forecast_set:
        list_of_predictions_using_lr.append(float(i))
    today_date = date.today()
    today_date_in_string = today_date.strftime("%d/%m/%Y")
    today_date_in_string_for_firebase = today_date.strftime("%d-%m-%Y")
    today_stock = today_stock.round(2)
    idea, decision = final_result(global_polarity, today_stock, mean)
    uploadARIMAfile('static/ARIMA/'+quote+'.png',
                    today_date_in_string_for_firebase, quote)
    uploadLSTMfile('static/LSTM/'+quote+'.png',
                   today_date_in_string_for_firebase, quote)
    uploadLINEARREGRESSIONfile(
        'static/LINEAR_REGRESSION/'+quote+'.png', today_date_in_string_for_firebase, quote)
    uploadTrendsFile('static/trends/'+quote+'.png',
                     today_date_in_string_for_firebase, quote)
    uploadSentimentChartsFile('static/SENTIMENT_CHARTS/' +
                              quote+'.png', today_date_in_string_for_firebase, quote)
    result_images = getImageLinkFromFirebase(quote)
    new_data = {
        today_date_in_string: {
            'LSTM_ERROR': error_lstm,
            'LSTM_PREDICTION': float(lstm_pred),
            'ARIMA_ERROR': error_arima,
            'ARIMA PREDICTION': float(arima_pred),
            'LINEAR_REGRESSSION_ERROR': error_linear_regression,
            'LINEAR_REGRESSION_PREDICTION': float(linear_regression_pred),
            'NEXT_SEVEN_DAYS_RESULTS': list_of_predictions_using_lr,
            'today_stock_results': {
                'today_open': float(today_stock['Open']),
                'today_close': float(today_stock['Close']),
                'today_high': float(today_stock['High']),
                'today_low': float(today_stock['Low']),
                'today_vol': float(today_stock['Volume']),
                'today_adj_close': float(today_stock['Adj Close'])
            },
            'PREDICTED_IMAGES': {
                'LSTM': result_images[1],
                'ARIMA': result_images[0],
                'LINEAR REGRESSION': result_images[2],
                'TREND': result_images[3],
                'SENTIMENT_CHART': result_images[4]
            },
            'RECENT_TWEETS': recent_tweets,
            'OVERALL_RESULT': tw_polarity,
            'IDEA': idea,
            'DECISION': decision,
        }
    }
    with open('search_results.json', 'r') as file:
        existed_data = json.load(file)
        if (quote in existed_data):
            existed_data[quote].append(new_data)
        else:
            existed_data[quote] = [new_data]

    with open('search_results.json', 'w') as file:
        json.dump(existed_data, file)

    return new_data[today_date_in_string]


def search(quote):

    with open('search_results.json', 'r') as file:
        existed_data = json.load(file)
    if (quote in existed_data):
        print("Found in json")
        stock_data = existed_data[quote]
        todays_date = date.today().strftime("%d/%m/%Y")
        today_date_present = False
        for i in stock_data:
            if todays_date in i:
                today_date_present = True
                required_result = i
                break

        if (today_date_present):
            print("Found today's date")
            return required_result[todays_date]

        else:
            required_result = common_ml_code(quote)
            return required_result

    else:
        required_result = common_ml_code(quote)
        return required_result


def get_today_stock_results(quote):
    end = dt.datetime.now()
    start = dt.date(end.year, end.month, end.day-2)
    data = yf.download(quote, start=start, end=end)
    today_stock = data.iloc[-1:]
    today_stock = today_stock.round(2)
    print(type(today_stock['Open']))

    print(today_stock)
    print(float(today_stock['Open']))
    temp = float(today_stock['Open'])
    required_result = {
        'today_open': temp,
        'today_close': float(today_stock['Close']),
        'today_high': float(today_stock['High']),
        'today_low': float(today_stock['Low']),
        'today_vol': float(today_stock['Volume']),
        'today_adj_close': float(today_stock['Adj Close'])
    }

    return required_result


# Function to preprocess tweet text

def preprocess_tweet(tweet):
    port_stem = PorterStemmer()
    tweet = re.sub('[^a-zA-Z]', ' ', tweet)  # Remove non-alphabetic characters
    tweet = tweet.lower()  # Convert text to lowercase
    tweet = tweet.split()  # Tokenize text
    tweet = [port_stem.stem(word) for word in tweet if word not in stopwords
             ]  # Remove stopwords and apply stemming
    tweet = ' '.join(tweet)
    return tweet


def get_sentiment_for_news(news):
    # Load the model
    loaded_model = pickle.load(open('trained_sentiment_model.sav', 'rb'))

    # load the vectorizer
    loaded_vectorizer = pickle.load(open('tfidf_vectorizer.sav', 'rb'))
    tweets = [news]
    preprocessed_tweets = [preprocess_tweet(tweet) for tweet in tweets]
    X_new = loaded_vectorizer.transform(preprocessed_tweets)

    # Make predictions
    predictions = loaded_model.predict(X_new)
    sentiment = ""
    for i, prediction in enumerate(predictions):
        if prediction == 0:
            sentiment = 'Negative'
        else:
            sentiment = 'Positive'
    required_result = {'sentiment': sentiment, 'news': news}
    return required_result


def get_next_seven_days_stock_results(quote):

    return {}


app = Flask(__name__)
app.static_folder = 'static'


@app.route('/get_today_stock_results', methods=['POST', 'GET'])
def get_today_stock_results_function():
    quote = request.args["quote"]
    required_result = get_today_stock_results(quote)
    return Response(status=200, content_type='application/json', headers={'content-type': 'application/json'}, response=json.dumps(required_result))


@app.route('/')
def temp():
    return render_template('temp.html')


@app.route('/search', methods=['POST', 'GET'])
def search_function():
    quote = request.args["quote"]
    required_result = search(quote)
    return Response(status=200, content_type='application/json', headers={'content-type': 'application/json'}, response=json.dumps(required_result))


@app.route('/get_next_seven_days_stock_results', methods=['POST', 'GET'])
def get_next_seven_days_results_function():
    quote = request.args["quote"]
    required_result = get_next_seven_days_stock_results(quote)
    return Response(status=200, content_type='application/json', headers={'content-type': 'application/json'}, response=json.dumps(required_result))


@app.route('/get_sentiment', methods=['POST', 'GET'])
def get_sentiment():
    news = request.args['news']
    required_result = get_sentiment_for_news(news)
    return Response(status=200, content_type='application/json', headers={'content-type': 'application/json'}, response=json.dumps(required_result))


if __name__ == '__main__':

    app.debug = True
    app.run(host='localhost', debug=True, port=3000)
    # folder_path = "C:\Users\lalit\Desktop\stock_price_prediction\Datasets"
