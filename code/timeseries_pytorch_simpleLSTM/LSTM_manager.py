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

###### making reproducable #######
# torch.set_deterministic(True)
# np.random.seed(0)
# torch.manual_seed(0)


################ CREATE THE PYTORCH LSTM MODEL ###################################
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=40, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size).cuda(), 
                            torch.zeros(1,1,self.hidden_layer_size).cuda())
        
    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]
      
class LSTMHandler():
    """
    A class that can train/save a model and make predictions.
    """
    def __init__(self):
        """
        Init class
        """
        self.data = None
        self.data_name = None
        self.train_data_normalized = None
        # use a train windows that is domain dependent here 365 since it is daily data per year
        self.train_window = 512
        self.test_data_size = None
        self.scaler = None
        self.device = None
        self.hist = None

    def create_train_test_data(self, data = None, data_name = None):
        # Create a new dataframe with only the 'Close column'
        self.data = data
        self.data_name = data_name
        all_data = self.data.values

        # Get the number of rows for test
        test_data_len = math.ceil(len(all_data)*0.25)

        self.test_data_size = test_data_len
        train_data = all_data[:-self.test_data_size]
        test_data = all_data[-self.test_data_size:]
        self.scaler = MinMaxScaler(feature_range=(-1, 1))

        print('train shape' , train_data.shape)
        print('test data shape' , test_data.shape)

        # reshape is necessa4ry because each row should be a sample so convert (132) -> (132,1)
        train_data_normalized = self.scaler.fit_transform(train_data.reshape(-1, 1))

        # maybe data normalization shoudl only be applied to training data and not on test data

        # Convert the data to a tensor
        self.train_data_normalized = torch.cuda.FloatTensor(train_data_normalized).view(-1)
        
    def create_trained_model(self, modelpath, epochs):
        # Set Device 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #assuming gpu is available
        print('device isssss', device)

        def create_inout_sequences(input_data, tw):
            inout_seq = []
            for i in range(len(input_data)-tw):
                train_seq = input_data[i:i+tw]
                train_label = input_data[i+tw:i+tw+1]
                inout_seq.append((train_seq, train_label))
            return inout_seq

        train_inout_seq = create_inout_sequences(self.train_data_normalized, self.train_window)

        model = LSTM().to(device)
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)
        
        ##### Train the model #####
        epochs = epochs
        print(epochs)
        self.hist = np.zeros(epochs)
        start_time = time.time()

        for i in range(epochs):
            for seq, labels in train_inout_seq:
                optimizer.zero_grad()
                model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).cuda(),
                                    torch.zeros(1, 1, model.hidden_layer_size).cuda())

                y_pred = model(seq)

                single_loss = loss_function(y_pred, labels)
                self.hist[i] = single_loss.item()

                single_loss.backward()
                optimizer.step()
            
            if i%5 == 1:
                print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

        training_time = time.time()-start_time
        print("Training time: {}".format(training_time))

        path_to_save = modelpath

        torch.save(model.state_dict(), path_to_save)
        return
    
    def make_predictions_from_model(self, modelpath):

        # Set Device 
        print("start predicting")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #assuming gpu is available
        print('device isssss', device)
        ####### making predictions #############
        fut_pred = self.test_data_size
        test_inputs = self.train_data_normalized[-self.train_window:].tolist()

        model = LSTM().to(device)
        model.load_state_dict(torch.load(modelpath))
        model.cuda()

        model.eval()
        print("model loaded")

        for i in range(fut_pred):
            seq = torch.cuda.FloatTensor(test_inputs[-self.train_window:])
            with torch.no_grad():
                model.hidden = (torch.zeros(1, 1, model.hidden_layer_size).cuda(), torch.zeros(1, 1, model.hidden_layer_size).cuda())
                modeloutput = model(seq).item()
                test_inputs.append(modeloutput)

        actual_predictions = self.scaler.inverse_transform(np.array(test_inputs[self.train_window:]).reshape(-1, 1))

        train = self.data[:-self.test_data_size]
        valid = self.data[-self.test_data_size:]
        valid['Predictions'] = actual_predictions

        # plt.figure(figsize=(16,8))
        # plt.title('Close Price History')

        # plt.plot(train[self.data_name])
        # plt.plot(valid[[self.data_name, 'Predictions']])

        # plt.xlabel('Date', fontsize=18)
        # plt.ylabel('Close Price USD ($)', fontsize=18)
        # plt.legend(['Train', 'Target', 'Predictions'], loc='lower right')

        # plt.show()

        # plt.figure(figsize=(16,8))
        # plt.title('Training Loss', fontsize=25)
        # plt.xlabel('Epoch', fontsize=18)
        # plt.ylabel('Loss', fontsize=18)
        # plt.show()

        y_pred = pd.Series(valid['Predictions'])
        y_pred.index = list(y_pred.index)
        print("predictions made")
        return y_pred
    
    def plot_training_error(self):
        plt.figure(figsize=(16,8))
        plt.title('Training Loss', fontsize=25)
        plt.xlabel('Epoch', fontsize=18)
        plt.ylabel('Loss', fontsize=18)
        plt.plot(self.hist)
        plt.show()
