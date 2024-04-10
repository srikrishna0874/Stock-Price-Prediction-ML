import datetime as dt
import math
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


end = dt.datetime.now()
start = dt.datetime(end.year-2, end.month, end.day)
df = yf.download('TSLA', start=start, end=end)

regressor = load_model("keras_model.h5")

dataset_train = df.iloc[0:int(0.8*len(df)), :]
dataset_test = df.iloc[int(0.8*len(df)):, :]

real_stock_price = dataset_test.iloc[:, 5:6].values

dataset_total = pd.concat(
    (dataset_train['Close'], dataset_test['Close']), axis=0)

testing_set = dataset_total[len(
    dataset_total) - len(dataset_test) - 7:].values
testing_set = testing_set.reshape(-1, 1)

sc = MinMaxScaler(feature_range=(0, 1))
testing_set = sc.fit_transform(testing_set)

X_test = []

for i in range(7, len(testing_set)):
    X_test.append(testing_set[i-7:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


predicted_stock_price = regressor.predict(X_test) # type: ignore
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
plt.plot(real_stock_price, label='Actual Price')
plt.plot(predicted_stock_price, 'r', label='Predicted Price')
plt.ylabel('Close Price')
plt.savefig('static/LSTM/{}.png'.format('TSLA'))
plt.close(fig)

error_lstm = math.sqrt(mean_squared_error(
    real_stock_price, predicted_stock_price))

forecasted_stock_price = regressor.predict(X_test) # type: ignore
forecasted_stock_price = sc.inverse_transform(forecasted_stock_price)

lstm_pred = forecasted_stock_price[0, 0]

print(lstm_pred, error_lstm)
