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

import thesis_experiments.experiment4.LSTM_manager_7D as LSTM_manager_7D

import optuna

import neptune
from neptunecontrib.api import log_chart
import os

neptune.init(project_qualified_name='mavantwout/Experiment-4',
             api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiODBkNzdjMDUtYmYxZi00ODFjLWExN2MtNjk3Y2MwZDE5N2U2In0=',
             )
def getDataInfo(datafilename, seed):
    split_on_data = datafilename.partition('/Dataset3/Missing')
    split_for_missing =  split_on_data[2].partition('/')
    missingness= split_for_missing[0]
    company = split_for_missing[2].split('_')[0]

    imputation = '5_imputations_combined'

    neptune.log_text('company', str(company))
    neptune.log_text('missingness', str(missingness))
    neptune.log_text('imputation', str(imputation))
    neptune.log_text('seed', str(seed))

PARAMS = {'epochs': 80,
        'lr':  0.004,
        'hls' : 90,
        'train_window': 380, 
        'opt' : 'SGD',
        'loss' : 'MSELoss',
        'dropout': 0.5,
        'num_layers': 2}

seeds = 5
for seed in range(seeds):
    # Create experiment
    neptune.create_experiment('7D_predictor_exp4_time_lag', params = PARAMS, upload_source_files=['./thesis_experiments/experiment4/LSTM_manager_7D.py', './thesis_experiments/experiment4/7D_predictor_time_lag.py'], tags=['single_run', '7D-prediction', '4-year', '20-step-ahead', '20-predictions', 'shifted30','time-lag','seed'+str(seed)])

    ############################  Single 20-step ahead prediction 6-D ########################## 
    FILEPATH = os.getenv('arg1')
    # FILEPATH = './thesis_datasets/Dataset3/Missing33/AAPL_Shifted_30ahead_m33.csv'
    getDataInfo(FILEPATH, seed)
    df = pd.read_csv(FILEPATH)	
    neptune.set_property('data', FILEPATH)

    data_name = 'Close'
    lagged_imputation_forwardfill ='Close_shifted_forwardfill' 
    lagged_imputation_globalmean ='Close_shifted_globalmean' 
    lagged_imputation_meanlast30 ='Close_shifted_meanlast260' 
    lagged_imputation_linearfit30 ='Close_shifted_linearfit260' 
    lagged_imputation_cubicfit30 ='Close_shifted_cubicfit260'
    time_lag = 'time_lag' 
    
    data = df.filter(items=[data_name, lagged_imputation_forwardfill, lagged_imputation_globalmean, lagged_imputation_meanlast30, lagged_imputation_linearfit30, lagged_imputation_cubicfit30, time_lag])

    test_size = 20
    # The test size here is 20, this creates the split between what data is known and not known, like training and test.
    y_train, y_test = temporal_train_test_split(data[data_name], test_size=test_size)

    y_train_lagged_forwardfill , y_test_lagged_forwardfill  = temporal_train_test_split(data[lagged_imputation_forwardfill], test_size=test_size)
    y_train_lagged_globalmean, y_test_lagged_globalmean = temporal_train_test_split(data[lagged_imputation_globalmean], test_size=test_size)
    y_train_lagged_meanlast30, y_test_lagged_meanlast30 = temporal_train_test_split(data[lagged_imputation_meanlast30], test_size=test_size)
    y_train_lagged_linearfit30, y_test_lagged_linearfit30 = temporal_train_test_split(data[lagged_imputation_linearfit30], test_size=test_size)
    y_train_lagged_cubicfit30, y_test_lagged_cubicfit30 = temporal_train_test_split(data[lagged_imputation_cubicfit30], test_size=test_size)


    s = LSTM_manager_7D.LSTMHandler(seed = seed)
    # the test size in this case is 1, since we are only trying to predict 1 value, but 20 steps ahead. 
    s.create_train_test_data(data = data,
     data_name = data_name,
     test_size=test_size
     )

    s.create_trained_model(params=PARAMS)

    # s.explain_simple_prediction()

    y_pred = s.make_predictions_from_model()

    y_test_to_predict = y_test[-test_size:]
    # y_test_to_predict = y_test[-(test_size-1):]
    #check if the y_test_to_predict is similar size to y_pred

    smape = smape_loss(y_test_to_predict, y_pred)
    neptune.log_metric('smape', smape)

    y_train = y_train
    #change this.
    fig, ax = plot_series(
        y_train,
        y_train_lagged_forwardfill,
        y_train_lagged_globalmean,
        y_train_lagged_meanlast30,
        y_train_lagged_linearfit30,
        y_train_lagged_cubicfit30,
        y_test, y_pred,
        labels=["y_train", "y_train_lagged_forwardfill","y_train_lagged_globalmean","y_train_lagged_meanlast30","y_train_lagged_linearfit30","y_train_lagged_cubicfit30","y_test", "y_pred"]
        )

    neptune.log_image('univariate_plot', fig)

    lossplot = s.plot_training_error()
    neptune.log_image('training_loss', lossplot)

    ax.get_legend().remove()
    log_chart(name='univariate_plot', chart=fig)

    neptune.stop()