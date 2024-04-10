import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.metrics import mean_squared_error
import datetime as dt
from statsmodels.tsa.arima.model import ARIMA
import math


def ARIMA_ALGORITHM(df, quote):
    uniqueVals = df["Code"].unique()
    len(uniqueVals)

    df = df.set_index("Code")

    def parser(x):
        return dt.datetime.strptime(x, '%Y-%m-%d')

    def arima_model(train, test):
        history = [x for x in train]
        predictions = list()
        for t in range(len(test)):
            model = ARIMA(history, order=(6, 1, 0))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
        return predictions
    

    for company in uniqueVals[:10]:
        data = (df.loc[company, :]).reset_index()
        data['Price'] = data['Close']
        Quantity_date = data[['Price', 'Date']]
        Quantity_date.index = Quantity_date['Date'].map(lambda x: parser(x))
        Quantity_date['Price'] = Quantity_date['Price'].map(lambda x: float(x))
        Quantity_date = Quantity_date.fillna(Quantity_date.bfill())
        Quantity_date = Quantity_date.drop(['Date'], axis=1)
        fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
        plt.plot(Quantity_date)
        plt.savefig('static/trends/'+quote+'.png')

        quantity = Quantity_date.values
        size = int(len(quantity)*0.8)
        train, test = quantity[0:size], quantity[size:len(quantity)]

        # fit in model
        predictions = arima_model(train, test)

        # plot graph
        fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
        plt.plot(test, 'g', label='Actual Price')
        plt.plot(predictions, 'r', label='Predicted Price')

        plt.savefig('static/ARIMA/'+quote+'.png')
        plt.close(fig)
        arima_pred = predictions[-2]
        print('tomorrow closing price is:', arima_pred)
        error_arima = math.sqrt(mean_squared_error(test, predictions))
        print('arima rmse', error_arima)

        return error_arima, arima_pred
