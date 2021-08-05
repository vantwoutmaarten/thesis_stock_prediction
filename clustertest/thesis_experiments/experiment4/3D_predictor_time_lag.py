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
from torch.random import seed

import thesis_experiments.experiment4.LSTM_manager_3D as LSTM_manager_3D

import optuna

import neptune
from neptunecontrib.api import log_chart
import os

neptune.init(project_qualified_name='mavantwout/Experiment-4',
             api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiODBkNzdjMDUtYmYxZi00ODFjLWExN2MtNjk3Y2MwZDE5N2U2In0=',
             )
def getDataInfo(datafilename, columnname, seed):
    split_on_data = datafilename.partition('/Dataset3/Missing')
    split_for_missing =  split_on_data[2].partition('/')
    missingness= split_for_missing[0]
    company = split_for_missing[2].split('_')[0]

    split_for_imputation= columnname.split('_')
    imputation = split_for_imputation[-1]

    neptune.log_text('company', str(company))
    neptune.log_text('missingness', str(missingness))
    neptune.log_text('imputation', str(imputation))
    neptune.log_text('seed', str(seed))

PARAMS = {'epochs': 25,
        'lr':  0.008,
        'hls' : 125,
        'train_window': 300, 
        'opt' : 'RMSprop',
        'loss' : 'MSELoss',
        'dropout': 0.1,
        'num_layers': 2}

seeds = 5
for seed in range(seeds):
    # Create experiment
    neptune.create_experiment('3D_predictor_exp4_time-lag', params = PARAMS, upload_source_files=['./thesis_experiments/experiment4/LSTM_manager_3D.py', './thesis_experiments/experiment4/3D_predictor_time_lag.py'], tags=['single_run', '3D-prediction', '4-year', '20-step-ahead', '20-predictions', 'shifted30', 'time-lag','seed'+str(seed)])

    ############################  Single 20-step ahead prediction 2-D ########################## 
    FILEPATH = os.getenv('arg1')
    columnname = os.getenv('arg2')
    # FILEPATH = './thesis_datasets/Dataset3/Missing33/AAPL_Shifted_30ahead_m33.csv'
    # columnname = 'Close_shifted_forwardfill'

    getDataInfo(FILEPATH, columnname, seed)
    df = pd.read_csv(FILEPATH)	
    neptune.set_property('data', FILEPATH)

    data_name = 'Close'
    lagged_data_name = columnname
    time_lag = 'time_lag'

    data = df.filter(items=[data_name, lagged_data_name, time_lag])


    test_size = 20
    # The test size here is 20, this creates the split between what data is known and not known, like training and test.
    y_train, y_test = temporal_train_test_split(data[data_name], test_size=test_size)
    y_train_lagged, y_test_lagged = temporal_train_test_split(data[lagged_data_name], test_size=test_size)

    s = LSTM_manager_3D.LSTMHandler(seed = seed)
    # the test size in this case is 1, since we are only trying to predict 1 value, but 20 steps ahead. 
    s.create_train_test_data(data = data, data_name = data_name, lagged_data_name=lagged_data_name, test_size=test_size)

    s.create_trained_model(params=PARAMS)

    y_pred = s.make_predictions_from_model()

    y_test_to_predict = y_test[-test_size:]
    # y_test_to_predict = y_test[-(test_size-1):]
    #check if the y_test_to_predict is similar size to y_pred

    smape = smape_loss(y_test_to_predict, y_pred)
    neptune.log_metric('smape', smape)

    y_train = y_train

    fig, ax = plot_series(y_train, y_train_lagged, y_test, y_pred, labels=["y_train", "y_train_lagged", "y_test", "y_pred"])

    neptune.log_image('univariate_plot', fig)

    lossplot = s.plot_training_error()
    neptune.log_image('training_loss', lossplot)

    ax.get_legend().remove()
    log_chart(name='univariate_plot', chart=fig)

    neptune.stop()