import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
import math


def LSTM_ALGORITHM(df,quote):

    # splitting data into training and testing data

    dataset_train = df.iloc[0:int(0.8*len(df)), :]
    dataset_test = df.iloc[int(0.8*len(df)):, :]

    # TO PREDICT STOCK PRICES OF NEXT N DAYS, STORE PREVIOUS N DAYS IN MEMORY WHILE TRAINING
    # HERE N=7

    training_set = df.iloc[:, 5:6].values
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    # Creating data stucture with 7 timesteps and 1 output.
    # 7 timesteps meaning storing trends from 7 days before current day to predict 1 next output

    X_train = []
    y_train = []

    for i in range(7, len(training_set_scaled)):
        X_train.append(training_set_scaled[i-7:i, 0])
        y_train.append(training_set_scaled[i, 0])
    # Convert list to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_forecast = np.array(X_train[-1, 1:])
    X_forecast = np.append(X_forecast, y_train[-1])
    # Reshaping: Adding 3rd dimension
    # .shape 0=row,1=col
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_forecast = np.reshape(X_forecast, (1, X_forecast.shape[0], 1))
    # For X_train=np.reshape(no. of rows/samples, timesteps, no. of cols/features)

    # building RNN
    # Initialise RNN
    regressor = Sequential()

    # Add first LSTM layer
    regressor.add(LSTM(units=50, return_sequences=True,
                       input_shape=(X_train.shape[1], 1)))
    # units=no. of neurons in layer
    # input_shape=(timesteps,no. of cols/features)
    # return_seq=True for sending recc memory. For last layer, retrun_seq=False since end of the line
    regressor.add(Dropout(0.1))

    # Add 2nd LSTM layer
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.1))

    # Add 3rd LSTM layer
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.1))

    # Add 4th LSTM layer
    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.1))

    # Add o/p layer
    regressor.add(Dense(units=1))

    # Compile
    regressor.compile(optimizer='adam', loss='mean_squared_error')

    # Training
    regressor.fit(X_train, y_train, epochs=25, batch_size=32)
    # For lstm, batch_size=power of 2

    real_stock_price = dataset_test.iloc[:, 5:6].values

    # To predict, we need stock prices of 7 days before the test set
    # So combine train and test set to get the entire data set

    dataset_total = pd.concat((dataset_train['Close'], dataset_test['Close']), axis=0)
    
    testing_set = dataset_total[len(dataset_total) - len(dataset_test) - 7:].values
    testing_set = testing_set.reshape(-1, 1)
    
    # Feature scaling
    testing_set = sc.transform(testing_set)

    # Create data structure
    X_test = []
    for i in range(7, len(testing_set)):
        X_test.append(testing_set[i-7:i, 0])
    # Convert list to numpy arrays
    X_test = np.array(X_test)

    # Reshaping: Adding 3rd dimension
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Testing Prediction
    predicted_stock_price = regressor.predict(X_test)

    # Getting original prices back from scaled values

    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    fig = plt.figure(figsize=(7.2, 4.8), dpi=65)

    plt.plot(real_stock_price, label='Actual Price')

    plt.plot(predicted_stock_price, 'r', label='Predicted Price')
    plt.ylabel('Close Price')

    plt.savefig('static/LSTM_'+quote+'.png')
    plt.close(fig)

    error_lstm = math.sqrt(mean_squared_error(
        real_stock_price, predicted_stock_price))

    # forecasting prediction

    forecasted_stock_price = regressor.predict(X_forecast)
    forecasted_stock_price = sc.inverse_transform(forecasted_stock_price)

    lstm_pred = forecasted_stock_price[0, 0]

    return lstm_pred, error_lstm
