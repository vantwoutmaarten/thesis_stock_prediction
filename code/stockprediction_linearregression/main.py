#Install the dependencies
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('bmh')

#store the data in a frame
df = pd.read_csv("./AAPL_Daily_Dec2016-Dec2020.csv")



# Visualize the close price data
plt.figure(figsize=(16,8))
plt.title('Apple')
plt.xlabel('Days')
plt.ylabel('Close price ($)')
plt.plot(df['Adj Close'])
# plt.show()

# get only close price
df = df[['Adj Close']]

#Create a variable to predict 'x' days out into the future
future_days = 50
#Create a new column (target) shifted x 'units/days up
df['Prediction'] = df[['Adj Close']].shift(-future_days)



# Create a feature data set (X) and convert it to a numpy array and remove the last 'x' rows/days
X = np.array(df.drop(['Prediction'], 1))[:-future_days]


# Create the target data set (y) and convert it to a numpy array and get all of the target values except the last 'x' rows
Y = np.array(df['Prediction'])[:-future_days]

# Split the data into 75% training and 25% testing
# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

# Create the linear regression model
lr = LinearRegression().fit(X, Y)

# Get the last 'x' rows of the feature dataset
x_future = df.drop(['Prediction'], 1)[:-future_days]
x_future = x_future.tail(future_days)
x_future = np.array(x_future)


# Show the model lr prediction
lr_prediction = lr.predict(x_future)
print(lr_prediction.shape)
predictions = lr_prediction

# Visualize the data
valid = df[X.shape[0]:]
valid['Predictions'] = predictions
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Days')
plt.ylabel('Close Price USD ($)')
plt.plot(df['Adj Close'])
plt.plot(valid[['Adj Close', 'Predictions']])
plt.legend(['original', 'Valid', 'predicted'])
plt.show()

