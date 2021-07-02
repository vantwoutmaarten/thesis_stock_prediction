import matplotlib
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
import optuna	
import neptune	
import neptunecontrib.monitoring.optuna as opt_utils



###### making reproducable #######	
import os	
os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'	
torch.set_deterministic(True)
torch.backends.cudnn.deterministic = True
# the detect anomaly is just for debugging, but sometimes it solves a problem of nvidia graphics drivers on windows.	
torch.autograd.set_detect_anomaly(True)	
## os.environ[‘CUDA_LAUNCH_BLOCKING’] = 1 if 1 all nothing happens asynchronizly, maybe it helps to debug something but should not be used in production I think. 	
# check influence on training time	
# os.environ[‘CUDA_LAUNCH_BLOCKING’] = 1	


#%%	
################ CREATE THE PYTORCH LSTM MODEL ###################################	
### 2 in 2 uit misschien even kijken hoe dat zit misschien willen we maar output size 1 hebben maar weet niet precies hoe de forward method daarmee omgaat. 	
### even in de docs kijken van PyTorch.	
### We are using this now and if it is working then we are changing the error calculation, by changin the train model part and the output size of the model to 1.	
class LSTM(nn.Module):	
    def __init__(self, input_size=6, hidden_layer_size=40, output_size=1, dropout = 0.0, num_layers=1):	
        super().__init__()
        self.hidden_layer_size = hidden_layer_size	
        self.lstm = nn.LSTM(input_size, hidden_layer_size, dropout=dropout, num_layers=num_layers)	
        self.linear = nn.Linear(hidden_layer_size, output_size)	
        self.hidden_cell = (torch.zeros(num_layers,1,self.hidden_layer_size).cuda(), 	
                            torch.zeros(num_layers,1,self.hidden_layer_size).cuda())	
        	
    def forward(self, input_seq):	
        # input_seq = input_seq.view(int(len(input_seq)/15), 1, 15)	

        #the 20 is the trainwindow instead of len(input_seq)
        input_seq = input_seq.view(len(input_seq), 1, 6)
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        #the 20 is the trainwindow instead of len(input_seq)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        # predictions = self.linear(lstm_out)	
        # return predictions[-1]	
        return predictions[-1,:].view(1,1)

    	
class LSTMHandler():	
    """	
    A class that can train/save a model and make predictions.	
    """	
    def __init__(self, seed = 0):	
        """	
        Init class	
        """	
        self.data = None	
        self.data_name = None	
        self.train_data_normalized = None	
        self.train_lagged_forwardfill= None
        self.train_lagged_globalmean= None
        self.train_lagged_meanlast30= None
        self.train_lagged_linearfit30= None
        self.train_lagged_cubicfit30= None

        self.lasttrainlabel = None
        # use a train windows that is domain dependent here 365 since it is daily data per year	
        self.train_window = None
        self.train_inout_seq = None
        self.test_data_size = None	
        self.scaler = None	
        self.device = None	
        self.hist = None	
        self.stateDict = None
        # Set the seed when intitializing this class.
        np.random.seed(seed)	
        torch.manual_seed(seed)	
        torch.cuda.manual_seed_all(seed)	

    def create_train_test_data(self, data = None,
     data_name = None,
     train_lagged_forwardfill=None,
     train_lagged_globalmean=None,
     train_lagged_meanlast30=None,
     train_lagged_linearfit30=None,
     train_lagged_cubicfit30=None,
     test_size=365
     ):	
        # Create a new dataframe with only the 'Close column'	
        self.data = data	
        self.data_name = data_name	
        self.train_lagged_forwardfill= train_lagged_forwardfill
        self.train_lagged_globalmean=train_lagged_globalmean
        self.train_lagged_meanlast30=train_lagged_meanlast30
        self.train_lagged_linearfit30=train_lagged_linearfit30
        self.train_lagged_cubicfit30=train_lagged_cubicfit30

        all_data = self.data


        # Get the number of rows for test by percentage	
        # test_data_len = math.ceil(len(all_data)*test_percentage)	
        # Get number of rows for test by number	
        test_data_len = test_size	
        self.test_data_size = test_data_len	

        self.lasttrainlabel = all_data[(-self.test_data_size-1):(-self.test_data_size)].iloc[0,0]

        # # create a differenced series
        def difference1lag(dataset):
            diff = list()
            for i in range(1, len(dataset)):
                value = dataset[i] - dataset[i - 1]
                diff.append(value)
            return pd.Series(diff)	
        
        all_data.iloc[:, 0] = difference1lag(all_data.iloc[:, 0])
        all_data.iloc[:, 1] = difference1lag(all_data.iloc[:, 1])
        all_data.iloc[:, 2] = difference1lag(all_data.iloc[:, 2])
        all_data.iloc[:, 3] = difference1lag(all_data.iloc[:, 3])
        all_data.iloc[:, 4] = difference1lag(all_data.iloc[:, 4])
        all_data.iloc[:, 5] = difference1lag(all_data.iloc[:, 5])
    
        train_data = all_data[:-(self.test_data_size)]	
        test_data = all_data[-self.test_data_size:]

        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        # self.scaler = StandardScaler()
        # 	
        print('train shape' , train_data.shape)	
        print('test data shape' , test_data.shape)
        # print('add test input shape' , add_test_input.shape)	
        # reshape is necessa4ry because each row should be a sample so convert (132) -> (132,1)	
        
        train_data_normalized = self.scaler.fit_transform(train_data)	
        # maybe data normalization shoudl only be applied to training data and not on test data	
        # Convert the data to a tensor	
        self.train_data_normalized = torch.cuda.FloatTensor(train_data_normalized).view(-1,6)


    def create_trained_model(self, params=None, modelpath=None):	
        # Set Device 	
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #assuming gpu is available	
        print('device isssss', device)	
        print(torch.cuda.get_device_name(0))	
        self.hidden_layer_size = params['hls']	
        self.train_window = params['train_window']	
        self.num_layers = params['num_layers']	
        	
        self.train_inout_seq = self._create_inout_sequences(self.train_data_normalized, self.train_window)	
        model = LSTM(hidden_layer_size=self.hidden_layer_size, num_layers = self.num_layers, dropout = params['dropout']).to(device)	
        loss_function = getattr(nn, params['loss'])()	
        optimizer = getattr(torch.optim, params['opt'])(model.parameters(), lr=params['lr'])	
        	
        ##### Train the model #####	
        epochs = params['epochs']	
        print(epochs)	
        self.hist = np.zeros(epochs)	
        start_time = time.time()	

        for i in range(epochs):	
            for seq, labels in self.train_inout_seq:	
                optimizer.zero_grad()	
                model.hidden_cell = (torch.zeros(self.num_layers, 1, model.hidden_layer_size).cuda(),	
                                    torch.zeros(self.num_layers, 1, model.hidden_layer_size).cuda())	
                y_pred = model(seq)	
                # y_pred = y_pred.view(1,2)	
                label = labels[0,0]
                single_loss = loss_function(y_pred, label)	
                # WHEN SHOULD THE LOSS BE AGGREGATED WITH mean()	
                # single_loss.backward(retain_graph=True)
                single_loss.backward()	
                optimizer.step()	
                	
            # neptune.log_metric('loss', single_loss)	
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
            # neptune.log_artifact(path_to_save)	
        return  self.stateDict	
    	
    def make_predictions_from_model(self, modelpath=None, modelstate=None):	
        ## The saved model path can be specified in Modelpath or the state of the model directly in ModelState	
        # Set Device 	
        print("start predicting")	
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #assuming gpu is available	
        print('device isssss', device)	
        	
        ####### making predictions #############	
        fut_pred = self.test_data_size	
        test_inputs = self.train_data_normalized[-(self.train_window+19):].tolist()	

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
            seq = torch.cuda.FloatTensor(test_inputs[(-self.train_window-19):-19])	
            with torch.no_grad():	
                model.hidden_cell = (torch.zeros(self.num_layers, 1, model.hidden_layer_size).cuda(), torch.zeros(self.num_layers, 1, model.hidden_layer_size).cuda())	
                modeloutput = model(seq).item()
                # test_inputs = torch.cat((test_inputs, modeloutput), 0)	
                test_inputs.append([modeloutput, np.NAN, np.NAN,np.NAN,np.NAN,np.NAN])	
        actual_predictions = self.scaler.inverse_transform(np.array(test_inputs[-self.test_data_size:]).reshape(-1, 6))	

        def invert_difference(dataset, lasttrainlabel):
            diff = list()
            value = lasttrainlabel
            for i in range(0, len(dataset)):
                value = value + dataset[i]
                diff.append(value)
            return pd.Series(diff)

        actual_predictions[:,0] = invert_difference(actual_predictions[:,0], self.lasttrainlabel)
                    
        train = self.data[:-self.test_data_size]		
        valid = self.data[-self.test_data_size:]		
        valid['Predictions'] = actual_predictions[:,0]	
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
        neptune_callback = opt_utils.NeptuneCallback(log_study=True, log_charts=True)	
        def func(trial):	
            tw = trial.suggest_int('tw', 20, 600)	
            ep = trial.suggest_int('ep', 1, 2)	
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
            y_pred = self.make_predictions_from_model(modelstate = trainedmodel)	
            smape = smape_loss(y_test, y_pred)	
            mape = mape_loss(y_test, y_pred)	
            neptune.log_metric('mape', mape)	
            return smape	
        study = optuna.create_study()	
        study.optimize(func, n_trials=2, callbacks=[neptune_callback])	
        opt_utils.log_study_info(study)

    # def explain_simple_prediction(self, modelpath=None, modelstate=None):

    #     # The SHAP deepexplainer needs a model and a background. Here the trained model is loaded.
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #assuming gpu is available	
    #     model = LSTM(hidden_layer_size=self.hidden_layer_size, num_layers = self.num_layers).to(device)	
    #     if(modelpath != None):	
    #         model.load_state_dict(torch.load(modelpath))	
    #     elif(modelstate != None):	
    #         model.load_state_dict(modelstate)	
    #     else:	
    #         model.load_state_dict(self.stateDict)	

    #     # The SHAP deepexplainer needs a model and a background. Here the background is created and 15 samples from the trainset are selected to calculate the expected value.
    #     training_windows = np.array([trainingwindow[0] for trainingwindow in self.train_inout_seq])
    #     background = training_windows[np.random.choice(training_windows.shape[0], 5, replace=False)]

    #     # training_windows = np.array([trainingwindow[0].cpu().detach().numpy() for trainingwindow in self.train_inout_seq])

    #     print(type(background))
    #     print(background.shape)

    #     background = background.tolist()
    #     # background = background.astype(float)
    #     print(type(background))
      
    #     background = torch.stack(background)
        
    #     print(type(background))

    #     # print(background.shape[0])
    #     # print(background.shape)
    #     # Here the deepexplainer is made using the trained model and the samples to come up with averages.
    #     print("create deepexplainer")
    #     e = shap.DeepExplainer(model, background.view(5,20,2))

    #     # To evaluate a prediction with the explainer test inputs should be specified.

    #     # THIS CHANGE IS MADE TO MAKE THE TESTINOUTSEQ BE FIT, BUT MAYBE WE DO NOT WANT THIS AND THE PREDICTION GETS CRAZY.
    #     test_inputs = self.train_data_normalized[-(self.train_window+20):]
    #     # test_inputs = self.train_data_normalized[-(self.train_window+19):].tolist()	
    #     self.test_inout_seq = self._create_inout_sequences(test_inputs, self.train_window)

    #     evaluation_windows = np.array([evaluation_window[0] for evaluation_window in self.test_inout_seq])
    #     evaluation_windows = evaluation_windows.tolist()
        
    #     print(type(evaluation_windows))
    #     # now the prediction(s) to explain is specified. 
    #     prediction_to_explain = evaluation_windows[0].view(1,20,2)

    #     print(type(prediction_to_explain))
    #     print(prediction_to_explain.shape)
        
    #     print("shap values")

    #     shap_values = e.shap_values(prediction_to_explain)

    #     shap.initjs()
    #     shap.force_plot(e.expected_value, shap_values, prediction_to_explain)
    #     # shap.text_plot(shap_values)
    #     return

    def _create_inout_sequences(self, input_data, tw):	
        inout_seq = []	
        for i in range(len(input_data)-tw-19):	
            train_seq = input_data[i:i+tw]	
            train_label = input_data[i+tw+19:i+tw+20]	
            inout_seq.append((train_seq, train_label))	
        return inout_seq	