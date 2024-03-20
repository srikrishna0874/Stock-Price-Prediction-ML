import math
import re
import os
import requests
import yfinance as yf
from flask import Flask, Response, render_template, redirect, send_file, url_for, request, send_from_directory
import datetime as dt
import pandas as pd
from Firebase.firebase import getImageLinkFromFirebase
from ML_models.lstm import *
from ML_models.arima import *
from ML_models.linear_regression import *
from Firebase import *
import json
from datetime import date


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
    lstm_pred, error_lstm = LSTM_ALGORITHM(df, quote)  # type: ignore
    error_arima, arima_pred = ARIMA_ALGORITHM(df, quote)  # type: ignore
    linear_regression_pred, error_linear_regression, forecast_set = LINEAR_REGRESSION_ALGORITHM(
        df, quote)
    list_of_predictions_using_lr = []
    for i in forecast_set:
        list_of_predictions_using_lr.append(float(i))
    today_date = date.today()
    today_date_in_string = today_date.strftime("%d/%m/%Y")
    today_stock = today_stock.round(2)
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
                'LINEAR REGRESSION': result_images[2]
            }
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
            return required_result

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


if __name__ == '__main__':

    app.debug = True
    app.run(host='localhost', debug=True, port=3000)
    # folder_path = "C:\Users\lalit\Desktop\stock_price_prediction\Datasets"