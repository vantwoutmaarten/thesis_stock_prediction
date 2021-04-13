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
    
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sktime.performance_metrics.forecasting import sMAPE, smape_loss, mape_loss
from sktime.forecasting.model_selection import temporal_train_test_split

# from soft_dtw_cuda import SoftDTW

import optuna
import neptune
import neptunecontrib.monitoring.optuna as opt_utils


###### making reproducable #######
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'

torch.set_deterministic(True)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

# the detect anomaly is just for debugging, but sometimes it solves a problem of nvidia graphics drivers on windows.
torch.autograd.set_detect_anomaly(True)
## os.environ[‘CUDA_LAUNCH_BLOCKING’] = 1 if 1 all nothing happens asynchronizly, maybe it helps to debug something but should not be used in production I think. 
# check influence on training time
# os.environ[‘CUDA_LAUNCH_BLOCKING’] = 1

#%%
################ CREATE THE PYTORCH LSTM MODEL ###################################
class LSTM(nn.Module):
    def __init__(self, input_size=2, hidden_layer_size=40, output_size=2, dropout = 0.0, num_layers=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        print('lstm input size', input_size)
        self.lstm = nn.LSTM(input_size, hidden_layer_size, dropout=dropout, num_layers=num_layers)
        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(num_layers,1,self.hidden_layer_size).cuda(), 
                            torch.zeros(num_layers,1,self.hidden_layer_size).cuda())
        
    def forward(self, input_seq):
        # input_seq = input_seq.view(int(len(input_seq)/15), 1, 15)
        input_seq = input_seq.view(len(input_seq), 1, 2)

        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)

        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        # predictions = self.linear(lstm_out)
        return predictions[-1,:]
      
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
        self.lagged_data_name = None
        self.train_data_normalized = None
        # use a train windows that is domain dependent here 365 since it is daily data per year
        self.train_window = None
        self.test_data_size = None
        self.scaler = None
        self.device = None
        self.hist = None
        self.stateDict = None

    def create_train_test_data(self, data = None, data_name = None, lagged_data_name=None, test_size=365):
        # Create a new dataframe with only the 'Close column'
        self.data = data
        self.data_name = data_name
        self.lagged_data_name = lagged_data_name
        all_data = self.data

        # Get the number of rows for test by percentage
        # test_data_len = math.ceil(len(all_data)*test_percentage)
        # Get number of rows for test by number
        test_data_len = test_size
        self.test_data_size = test_data_len

        train_data = all_data[:-self.test_data_size]
        test_data = all_data[-self.test_data_size:]
        # self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaler = StandardScaler()

        print('train shape' , train_data.shape)
        print('test data shape' , test_data.shape)

        # reshape is necessa4ry because each row should be a sample so convert (132) -> (132,1)
        train_data_normalized = self.scaler.fit_transform(train_data)

        # maybe data normalization shoudl only be applied to training data and not on test data

        # Convert the data to a tensor
        self.train_data_normalized = torch.cuda.FloatTensor(train_data_normalized).view(-1,2)
        
    def create_trained_model(self, params=None, modelpath=None):
        # Set Device 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #assuming gpu is available
        print('device isssss', device)
        print(torch.cuda.get_device_name(0))

        self.hidden_layer_size = params['hls']
        self.train_window = params['train_window']
        self.num_layers = params['num_layers']
        
        def create_inout_sequences(input_data, tw):
            inout_seq = []
            for i in range(len(input_data)-tw):
                train_seq = input_data[i:i+tw]
                train_label = input_data[i+tw:i+tw+1]
                inout_seq.append((train_seq, train_label))
            return inout_seq

        
        train_inout_seq = create_inout_sequences(self.train_data_normalized, self.train_window)

        model = LSTM(hidden_layer_size=self.hidden_layer_size, num_layers = self.num_layers, dropout = params['dropout']).to(device)
        loss_function = getattr(nn, params['loss'])()


        optimizer = getattr(torch.optim, params['opt'])(model.parameters(), lr=params['lr'])
        
        ##### Train the model #####
        epochs = params['epochs']
        print(epochs)
        self.hist = np.zeros(epochs)
        start_time = time.time()

        for i in range(epochs):
            for seq, labels in train_inout_seq:
                optimizer.zero_grad()
                model.hidden_cell = (torch.zeros(self.num_layers, 1, model.hidden_layer_size).cuda(),
                                    torch.zeros(self.num_layers, 1, model.hidden_layer_size).cuda())

                y_pred = model(seq)
                y_pred = y_pred.view(1,2)
                single_loss = loss_function(y_pred, labels)

                # WHEN SHOULD THE LOSS BE AGGREGATED WITH mean()
                single_loss.backward(retain_graph=True)
                optimizer.step()
                
            neptune.log_metric('loss', single_loss)
            self.hist[i] = single_loss.item()
            
            # if i%1 == 1:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

        training_time = time.time()-start_time
        print("Training time: {}".format(training_time))

        self.stateDict = model.state_dict()
        if(modelpath!=None):
            path_to_save = modelpath
            torch.save(self.stateDict, path_to_save)
            neptune.log_artifact(path_to_save)

        return  self.stateDict
    
    def make_predictions_from_model(self, modelpath=None, modelstate=None):
        ## The saved model path can be specified in Modelpath or the state of the model directly in ModelState
        # Set Device 
        print("start predicting")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #assuming gpu is available
        print('device isssss', device)
        
        ####### making predictions #############
        fut_pred = self.test_data_size
        test_inputs = self.train_data_normalized[-self.train_window:]

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
                model.hidden_cell = (torch.zeros(self.num_layers, 1, model.hidden_layer_size).cuda(), torch.zeros(self.num_layers, 1, model.hidden_layer_size).cuda())
                modeloutput = model(seq).view(1,2)
                test_inputs = torch.cat((test_inputs, modeloutput), 0)
                
                # test_inputs.append(modeloutput)

        actual_predictions = self.scaler.inverse_transform(np.array((test_inputs[self.train_window:]).reshape(-1, 2).cpu()))

        train = self.data[:-self.test_data_size]
        valid = self.data[-self.test_data_size:]
        valid['Prediction'] = actual_predictions[:,0]

        y_pred = pd.Series(valid['Prediction'])
        y_pred.index = list(y_pred.index)
        print("predictions made")
        valid['Prediction_Lag'] = actual_predictions[:,1]

        y_pred_lag = pd.Series(valid['Prediction_Lag'])
        y_pred_lag.index = list(y_pred.index)

        return y_pred, y_pred_lag
    
    def plot_training_error(self):
        fig = plt.figure(figsize=(16,8))
        plt.title('Training Loss', fontsize=25)
        plt.xlabel('Epoch', fontsize=18)
        plt.ylabel('Loss', fontsize=18)
        plt.plot(self.hist)
        # plt.show()
        return fig

    def optimize(self):
        y = self.data[self.data_name]
        y_train, y_test = temporal_train_test_split(y, test_size=self.test_data_size)

        neptune_callback = opt_utils.NeptuneCallback()

        def func(trial):
            tw = trial.suggest_int('tw', 20, 600)
            ep = trial.suggest_int('ep', 4, 25)
            lr = trial.suggest_uniform('lr', 0.00001, 0.01)
            hls = trial.suggest_int('hls', 1, 100)
            opt = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]) 
            loss = trial.suggest_categorical("loss", ["MSELoss"]) 
            stackedlayers = trial.suggest_int('stacked', 1, 2)
            dropout = trial.suggest_uniform('dropout', 0.0, 0.65)

            PARAMS = {'epochs': ep,
            'lr': lr,
            'hls' : hls,
            'train_window': tw, 
            'opt' : opt,
            'loss' : loss,
            'dropout': dropout,
            'num_layers': stackedlayers}

            trainedmodel = self.create_trained_model(params=PARAMS)

            # In the training the loss of multiple time series are included, since the predictions depend on eachother, but for the optimization we only want the main series,
            # the stock price to perform well.
            y_pred, y_pred_lag = self.make_predictions_from_model(modelstate = trainedmodel)
            smape = smape_loss(y_test, y_pred)
            mape = mape_loss(y_test, y_pred)
            neptune.log_metric('mape', mape)

            return smape

        study = optuna.create_study()

        study.optimize(func, n_trials=50, callbacks=[neptune_callback])
        opt_utils.log_study_info(study)

        print(study.best_params)