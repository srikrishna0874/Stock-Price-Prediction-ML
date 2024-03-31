import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import datetime as dt
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt2
import math


def LINEAR_REGRESSION_ALGORITHM(df, quote):
    forecast_out = int(7)

    # Price after n days
    df['Close after n days'] = df['Close'].shift(-forecast_out)

    # remove NaN
    df_new = df[['Close', 'Close after n days']]

    # discard last 35 rows and reshape
    y = np.array(df_new.iloc[:-forecast_out, -1])
    y = np.reshape(y, (-1, 1))

    X = np.array(df_new.iloc[:-forecast_out, 0:-1])
    X_to_be_forecasted = np.array(df_new.iloc[-forecast_out:, 0:-1])

    # Traning, testing to plot graphs, check accuracy
    X_train = X[0:int(0.8*len(df)), :]
    X_test = X[int(0.8*len(df)):, :]
    y_train = y[0:int(0.8*len(df)), :]
    y_test = y[int(0.8*len(df)):, :]

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    X_to_be_forecasted = sc.transform(X_to_be_forecasted)
    # Training

    clf = LinearRegression(n_jobs=-1)
    clf.fit(X_train, y_train)

    # testing
    y_test_pred = clf.predict(X_test)
    y_test_pred = y_test_pred*(1.04)

    fig = plt2.figure(figsize=(7.2, 4.8), dpi=65)

    plt2.plot(y_test, label='Actual Price')
    plt2.plot(y_test_pred, 'r', label='Predicted Price')

    plt2.savefig('static/LINEAR_REGRESSION/'+quote+'.png')
    plt2.close(fig)

    error_linear_regression = math.sqrt(
        mean_squared_error(y_test, y_test_pred))
    # forecasting

    forecast_set = clf.predict(X_to_be_forecasted)
    forecast_set = forecast_set*(1.04)
    mean = forecast_set.mean()
    linear_regression_pred = forecast_set[0, 0]

    return linear_regression_pred, error_linear_regression, forecast_set
