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

from sktime.performance_metrics.forecasting import sMAPE, smape_loss
from sktime.forecasting.model_selection import temporal_train_test_split

import optuna

###### making reproducable #######
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'

torch.set_deterministic(True)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

################ CREATE THE PYTORCH LSTM MODEL ###################################
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=40, output_size=1, dropout = 0.0, num_layers=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size, dropout=dropout, num_layers=num_layers)
        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(num_layers,1,self.hidden_layer_size).cuda(), 
                            torch.zeros(num_layers,1,self.hidden_layer_size).cuda())
        
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
        self.train_window = None
        self.test_data_size = None
        self.scaler = None
        self.device = None
        self.hist = None
        self.stateDict = None

    def create_train_test_data(self, data = None, data_name = None, test_size=365):
        # Create a new dataframe with only the 'Close column'
        self.data = data
        self.data_name = data_name
        all_data = self.data.values

        # Get the number of rows for test by percentage
        # test_data_len = math.ceil(len(all_data)*test_percentage)
        # Get number of rows for test by number
        test_data_len = test_size
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
        
    def create_trained_model(self, modelpath=None, epochs= 10, lr = 0.0005, hidden_layer_size = 40, train_window=365, optimizer_name="Adam", loss_name="MSELoss", num_layers = 1, dropout = 0.0):
        # Set Device 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #assuming gpu is available
        print('device isssss', device)

        self.hidden_layer_size = hidden_layer_size
        self.train_window = train_window
        self.num_layers = num_layers
        
        def create_inout_sequences(input_data, tw):
            inout_seq = []
            for i in range(len(input_data)-tw):
                train_seq = input_data[i:i+tw]
                train_label = input_data[i+tw:i+tw+1]
                inout_seq.append((train_seq, train_label))
            return inout_seq

        
        train_inout_seq = create_inout_sequences(self.train_data_normalized, self.train_window)

        model = LSTM(hidden_layer_size=hidden_layer_size, num_layers = self.num_layers, dropout = dropout).to(device)
        loss_function = getattr(nn, loss_name)()


        optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
        
        ##### Train the model #####
        epochs = epochs
        print(epochs)
        self.hist = np.zeros(epochs)
        start_time = time.time()

        for i in range(epochs):
            for seq, labels in train_inout_seq:
                optimizer.zero_grad()
                model.hidden_cell = (torch.zeros(num_layers, 1, model.hidden_layer_size).cuda(),
                                    torch.zeros(num_layers, 1, model.hidden_layer_size).cuda())

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

        self.stateDict = model.state_dict()
        if(modelpath!=None):
            path_to_save = modelpath
            torch.save(self.stateDict, path_to_save)

        return  self.stateDict
    
    def make_predictions_from_model(self, modelpath=None, modelstate=None):
        ## The saved model path can be specified in Modelpath or the state of the model directly in ModelState

        # Set Device 
        print("start predicting")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #assuming gpu is available
        print('device isssss', device)
        ####### making predictions #############
        fut_pred = self.test_data_size
        test_inputs = self.train_data_normalized[-self.train_window:].tolist()

        model = LSTM(hidden_layer_size=self.hidden_layer_size, num_layers = self.num_layers).to(device)
        if(modelpath != None):
            model.load_state_dict(torch.load(modelpath))
        elif(modelstate != None):
            model.load_state_dict(modelstate)
        else:
            model.load_state_dict(self.stateDict)

        model.cuda()
        model.eval()
        print("model loaded")

        for i in range(fut_pred):
            seq = torch.cuda.FloatTensor(test_inputs[-self.train_window:])
            with torch.no_grad():
                model.hidden = (torch.zeros(self.num_layers, 1, model.hidden_layer_size).cuda(), torch.zeros(self.num_layers, 1, model.hidden_layer_size).cuda())
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

    def optimize(self):
        y = self.data[self.data_name]
        y_train, y_test = temporal_train_test_split(y, test_size=self.test_data_size)

        def func(trial):
            tw = trial.suggest_int('tw', 20, 600)
            ep = trial.suggest_int('ep', 4, 18)
            lr = trial.suggest_uniform('lr', 0.00001, 0.01)
            hls = trial.suggest_int('hls', 1, 100)
            opt = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]) 
            loss = trial.suggest_categorical("loss", ["MSELoss", "KLDivLoss"]) 
            stackedlayers = trial.suggest_int('stacked', 1, 4)
            dropout = trial.suggest_uniform('dropout', 0.0, 0.65)

            trainedmodel = self.create_trained_model(epochs=ep, lr = lr, hidden_layer_size = hls, train_window=tw, optimizer_name=opt, loss_name = loss, num_layers=stackedlayers, dropout = dropout)

            y_pred = self.make_predictions_from_model(modelstate = trainedmodel)
            smape = smape_loss(y_test, y_pred)

            return smape

        study = optuna.create_study()

        study.optimize(func, n_trials=50)

        print(study.best_params)