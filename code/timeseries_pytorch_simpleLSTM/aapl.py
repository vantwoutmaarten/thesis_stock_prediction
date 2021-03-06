# %%
import torch
import torch.nn as nn
import math
import time

import pandas_datareader as web
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
    
from sklearn.preprocessing import MinMaxScaler

# Set Device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #assuming gpu is available
print('device isssss', device)

#Get the stock quote
# df = web.DataReader('AAPL', data_source='yahoo', start='2016-01-01',  end='2020-11-09')
# data = df.filter(['Close'])

df = pd.read_csv("./synthetic_data/brownian_scenarios/upward_mu_021_sig_065.csv")
data = df.filter(['stockprice'])


#%%
#Visualize the closing price history
# plt.figure(figsize=(16,8))
# plt.title('Close Price History')
# plt.plot(df['Close'])
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Close Price USD ($)', fontsize=18)
# plt.show()

# Create a new dataframe with only the 'Close column'


all_data = data.values

print('data info: ', data.info)


# Get the number of rows for test
test_data_len = math.ceil(len(all_data)*0.25)

test_data_size = test_data_len
train_data = all_data[:-test_data_size]
test_data = all_data[-test_data_size:]
scaler = MinMaxScaler(feature_range=(-1, 1))

print('train shape' , train_data.shape)
print('test data shape' , test_data.shape)

# reshape is necessa4ry because each row should be a sample so convert (132) -> (132,1)
train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 1))

# maybe data normalization shoudl only be applied to training data and not on test data

# Convert the data to a tensor
train_data_normalized = torch.cuda.FloatTensor(train_data_normalized).view(-1)

# %%
# use a train windows that is domain dependent here 12, since montly data
train_window = 365

def create_inout_sequences(input_data, tw):
    inout_seq = []
    for i in range(len(input_data)-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq, train_label))
    return inout_seq

train_inout_seq = create_inout_sequences(train_data_normalized, train_window)

# %%
################ CREATE THE PYTORCH LSTM MODEL ###################################
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size), 
                            torch.zeros(1,1,self.hidden_layer_size))
        
    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

model = LSTM().to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# %%
##### Train the model #####

epochs = 3
hist = np.zeros(epochs)
start_time = time.time()

for i in range(epochs):
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).cuda(),
                            torch.zeros(1, 1, model.hidden_layer_size).cuda())

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        hist[i] = single_loss.item()

        single_loss.backward()
        optimizer.step()
    
    if i%25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

training_time = time.time()-start_time
print("Training time: {}".format(training_time))
# %%
####### making predictions #############
fut_pred = test_data_size
test_inputs = train_data_normalized[-train_window:].tolist()

model.eval()

for i in range(fut_pred):
    seq = torch.cuda.FloatTensor(test_inputs[-train_window:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size).cuda(), torch.zeros(1, 1, model.hidden_layer_size).cuda())
        modeloutput = model(seq).item()
        test_inputs.append(modeloutput)

actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:]).reshape(-1, 1))

train = data[:-test_data_size]
valid = data[-test_data_size:]
valid['Predictions'] = actual_predictions

plt.figure(figsize=(16,8))
plt.title('Close Price History')

# plt.plot(train['Close'])
# plt.plot(valid[['Close', 'Predictions']])
plt.plot(train['stockprice'])
plt.plot(valid[['stockprice', 'Predictions']])

plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.legend(['Train', 'Target', 'Predictions'], loc='lower right')

plt.show()

plt.figure(figsize=(16,8))
plt.title('Training Loss', fontsize=25)
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Loss', fontsize=18)
plt.plot(hist)
plt.show()
# %%