# %%
from sktime.forecasting.model_selection import (
    ForecastingGridSearchCV,
    SlidingWindowSplitter,
    temporal_train_test_split,
)
import numpy as np
import pandas as pd
from sktime.utils.plotting import plot_series
from sktime.performance_metrics.forecasting import sMAPE, smape_loss

import LSTM_manager_2D_20ahead_20pred

import optuna

import neptune
from neptunecontrib.api import log_chart
import os

neptune.init(project_qualified_name='mavantwout/Stocks',
             api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiODBkNzdjMDUtYmYxZi00ODFjLWExN2MtNjk3Y2MwZDE5N2U2In0=',
             )

# PARAMS = {'epochs': 44,	
#     'lr': 0.007756,	
#     'hls' : 33,	
#     'train_window': 494, 	
#     'opt' : 'Adam',	
#     'loss' : 'MSELoss',	
#     'dropout': 0.0,	
#     'num_layers': 1}

# Create experiment	
# neptune.create_experiment('2D_20-step ahead prediction_seed1', params = PARAMS, upload_source_files=['./LSTM_manager_2D_20ahead_20pred.py', './2D_predictor_20ahead_20pred.py'], tags=['single_run', '2D-prediction', '4-year', '20-step-ahead', '20-predictions'])

# ############################  Single 20-step ahead prediction 2-D ########################## 	
# FILEPATH = os.getenv('arg1')		
# df = pd.read_csv(FILEPATH)		
# neptune.set_property('data', FILEPATH)	
# data_name = 'Close'	
# lagged_data_name = 'Close_ahead30'	
# data = df.filter(items=[data_name, lagged_data_name])	
# test_size = 20	

# # The test size here is 20, this creates the split between what data is known and not known, like training and test.	
# y_train, y_test = temporal_train_test_split(data[data_name], test_size=test_size)	
# y_train_lagged, y_test_lagged = temporal_train_test_split(data[lagged_data_name], test_size=test_size)	
# s = LSTM_manager_2D_20ahead_20pred.LSTMHandler()

# # the test size in this case is 1, since we are only trying to predict 1 value, but 20 steps ahead. 	
# s.create_train_test_data(data = data, data_name = data_name, lagged_data_name=lagged_data_name, test_size=test_size)	
# s.create_trained_model(params=PARAMS)	
# y_pred = s.make_predictions_from_model()	
# y_test_to_predict = y_test[-test_size:]	

# # y_test_to_predict = y_test[-(test_size-1):]	
# #check if the y_test_to_predict is similar size to y_pred	
# smape = smape_loss(y_test_to_predict, y_pred)	
# neptune.log_metric('smape', smape)	

# y_train = y_train	
# fig, ax = plot_series(y_train, y_train_lagged, y_test, y_test_lagged, y_pred, labels=["y_train", "y_train_lagged", "y_test", "y_test_lagged", "y_pred"])	
# neptune.log_image('univariate_plot', fig)	

# lossplot = s.plot_training_error()	
# neptune.log_image('training_loss', lossplot)	
# ax.get_legend().remove()	
# log_chart(name='univariate_plot', chart=fig)	

# neptune.stop()

############## OPTIMIZER ##################	
# Create experiment	
neptune.create_experiment('2D_20-step ahead prediction_optimization_test', upload_source_files=['./LSTM_manager_2D_20ahead_20pred.py', '2D_predictor_20ahead_20pred.py'], tags=['optimization', 'single_run', '2D-prediction', '4-year', '20-step-ahead', '20-predictions', 'shifted6'])	
 ############################  Single 20-step ahead prediction 2-D ########################## 	
FILEPATH = os.getenv('arg1')		
df = pd.read_csv(FILEPATH)		
neptune.set_property('data', FILEPATH)	

data_name = 'noisy_sin'
lagged_data_name = 'noisy_sin_ahead6'
data = df.filter(items=[data_name, lagged_data_name])


data = df.filter(items=[data_name, lagged_data_name])	
# The test size here is 20, this creates the split between what data is known and not known, like training and test.	
y_train, y_test = temporal_train_test_split(data[data_name], test_size=20)	
y_train_lagged, y_test_lagged = temporal_train_test_split(data[lagged_data_name], test_size=20)	
s = LSTM_manager_2D_20ahead_20pred.LSTMHandler()	
# the test size in this case is 1, since we are only trying to predict 1 value, but 20 steps ahead. 	
s.create_train_test_data(data = data, data_name = data_name, lagged_data_name=lagged_data_name, test_size=20)	
s.optimize()	
neptune.stop()