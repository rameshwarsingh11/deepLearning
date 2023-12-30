import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,1:2].values

sc = MinMaxScaler(feature_range= (0,1))
training_set_scalaed = sc.fit_transform(training_set)

x_train = []
y_train = []

for i in range(60,1258):
  x_train.append(training_set_scalaed[i-60:i,0])
  y_train.append(training_set_scalaed[i,0])

x_train,y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

regressor = Sequential()
regressor.add(LSTM(units = 50, return_sequences= True, input_shape = (x_train.shape[1],1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences= True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences= True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))


regressor.add(Dense(units = 1))

#compiling the RNN code
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#Model fitting code
regressor.fit(x_train,y_train,epochs=100, batch_size=32)

dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:,1:2].values

dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']), axis =0)
inputs = dataset_total[len(dataset_total)-len(dataset_test)- 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

x_test = []

for i in range(60,80):
  x_test.append(inputs[i-60:i,0])

x_test = np.array(x_test)

x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

predicted_stock_price= regressor.predict(x_test)

predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Predictions for 2020')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()