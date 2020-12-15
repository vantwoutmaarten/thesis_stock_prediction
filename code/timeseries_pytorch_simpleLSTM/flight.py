# %%
import torch
import torch.nn as nn

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

flight_data = sns.load_dataset("flights")

plt.figure(figsize=(15,5))
plt.title('Month vs Passenger')
plt.ylabel('Total Passengers')
plt.grid(True)
# autoscale for cutting of input without data
plt.autoscale(axis='x', tight=True)
plt.plot(flight_data['passengers'])
plt.show()

all_data = flight_data['passengers'].values.astype(float)

test_data_size = 20
train_data = all_data[:-test_data_size]
test_data = all_data[-test_data_size:]
scaler = MinMaxScaler(feature_range=(-1, 1))

# reshape is necessa4ry because each row should be a sample so convert (132) -> (132,1)
train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 1))

# maybe data normalization shoudl only be applied to training data and not on test data

# Convert the data to a tensor
train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
# use a train windows that is domain dependent here 12, since montly data
train_window = 25

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

model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# %%
##### Train the model #####

epochs = 1

for i in range(epochs):
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()
    
    if i%25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
# %%
####### making predictions #############
fut_pred = 20
test_inputs = train_data_normalized[-train_window:].tolist()

model.eval()

for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[-train_window:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size), torch.zeros(1, 1, model.hidden_layer_size))
        modeloutput = model(seq).item()
        print(modeloutput)
        test_inputs.append(modeloutput)

actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:]).reshape(-1, 1))


x = np.arange(124, 144, 1)

plt.figure(figsize=(15,5))
plt.title('Month vs Passenger')
plt.ylabel('Total Passengers')
plt.grid(True)
# autoscale for cutting of input without data
plt.autoscale(axis='x', tight=True)
plt.plot(flight_data['passengers'])
plt.plot(x, actual_predictions)
plt.show()


# %%
