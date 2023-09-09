import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def draw():
    plt.plot(real_stock_price, color='red', label='Real Bitcoin Stock Price')
    plt.plot(predicted_stock_price, color='blue', label='Predicted Bitcoin Stock Price')
    plt.title('Bitcoin Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Bitcoin Stock Price')
    plt.legend()
    plt.show()


#read data
dataset_train = pd.read_csv('coin_Bitcoin.csv')
training_set = dataset_train.iloc[:, 6:7].values

#normalization
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

X_train = []
y_train = []
for i in range(60, 2963):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

regressor = Sequential()
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1))
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

real_stock_price = dataset_train.iloc[2991 - 100:,6:7].values
inputs = dataset_train.iloc[2991 - 100 - 60:, 6:7].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)


X_test = []
for i in range(60, 160):
  X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = X_test.astype(np.float32)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))




predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

draw()
#predict results


mse_test = mean_squared_error(real_stock_price, predicted_stock_price)
mae_test = mean_absolute_error(real_stock_price, predicted_stock_price)
r2_test = r2_score(real_stock_price, predicted_stock_price)

print(f'Mean Squared Error (MSE) on Test Data: {mse_test:.4f}')
print(f'Mean Absolute Error (MAE) on Test Data: {mae_test:.4f}')
print(f'R-squared (R2) on Test Data: {r2_test:.4f}')
