# Description: This program uses an artificial recurrent neural network (LSTM) with Keras
#               to predict the closing stock price of Apple using the past 60 day stock price.(1 feature only closing price)
# %%
# Import the libraries
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


#Get the stock quote
df = web.DataReader('AAPL', data_source='yahoo', start='2016-01-01',  end='2020-11-09')

print(df)

#Visualize the closing price history
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()

# Create a new dataframe with only the 'Close column'
data = df.filter(['Close'])
# Convert the dataframe to a numpy array
dataset = data.values
# Get the number of rows for training
training_data_len = math.ceil(len(dataset)*0.8)

print(training_data_len)

# Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# Create the training data set
# Create the scaled training data set
train_data = scaled_data[0:training_data_len , :]
# Split the data into x_train and y_train
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

# Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data, because lstm expects 3d first num_samples, num_timesteps, num_features
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

################# KERAS ####################
#Build the LSTM  model
model = Sequential()
# input shape is num_time_steps and num_features
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
# return_sequences is now false because no more LSTM layers will be used
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# train the model, fit is other word for train
model.fit(x_train, y_train, batch_size=1, epochs=1)

# %%
# Create the testing data set
# Create a new array containing scaled values from index 1543 to 2003
test_data = scaled_data[training_data_len-60:, :]
# Create the data sets x_test and y_test --------- the x values are scaled but the y values are the actual values
x_test = []
y_test = dataset[training_data_len:,:]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

# Convert the data to a numpy array
x_test, y_test = np.array(x_test), np.array(y_test)

print(x_test.shape)
# Reshape the data, because lstm expects 3d first num_samples, num_timesteps, num_features (just the close price)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get the model predicted price values for x_test
predictions = model.predict(x_test)

# with the scaler inverse transform we unscale the predicted values (we want the predictions to be y_test)
predictions = scaler.inverse_transform(predictions)
# %%
#Get the root mean squared error (RMSE)
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))

rmse

train = data[:training_data_len]
valid = data[training_data_len:]

valid['Predictions'] = predictions

# Visualize the data
plt.figure(figsize=(16,8))
plt.title('Apple stock price model prediction')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)')
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Target', 'Predictions'], loc='lower right')
plt.show()

# %%
# Now PREDICT VALUE FOR THE NEXT DAY
apple_quote_df = web.DataReader('AAPL', data_source='yahoo', start='2016-01-01',  end='2020-11-09')
# Create a new dataframenew_df = apple_quote_df.filter(['Close'])
# Get the last 60 day closing price values and convert to array
last_60_days = new_df[-60:].values
# Scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)
# Create the input
X_test = []
X_test.append(last_60_days_scaled)
#Convert the X_test data set to a numpy array
X_test = np.array(X_test)
# Reshape the data 
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# Get the predicted scaled price
pred_price = model.predict(X_test)
# Unscale the predictions
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)

# get the apple quote for the day after
apple_quote_2 = web.DataReader('AAPL', data_source='yahoo', start='2020-11-10',  end='2020-11-10')
print(apple_quote_2['Close'])
# %%
