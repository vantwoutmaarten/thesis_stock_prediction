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
from sklearn.preprocessing import MinMaxScaler
from torch.random import seed

import LSTM_manager_1D


import optuna

import neptune
from neptunecontrib.api import log_chart
import os

neptune.init(project_qualified_name='mavantwout/thesis-experiment-test',
             api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiODBkNzdjMDUtYmYxZi00ODFjLWExN2MtNjk3Y2MwZDE5N2U2In0=',
             )
def getDataInfo(datafilename, seed):
    split_on_data = datafilename.partition('/Dataset1/')
    split_for_missing =  split_on_data[2].partition('.')
    company = split_for_missing[0]

    imputation = 'no_imputation'

    print(company)
    print(imputation)

    neptune.log_text('company', str(company))
    neptune.log_text('imputation', str(imputation))
    neptune.log_text('seed', str(seed))

PARAMS = {'epochs': 2,
        'lr':  0.00029177560092619997,
        'hls' : 28,
        'train_window': 20, 
        'opt' : 'RMSprop',
        'loss' : 'MSELoss',
        'dropout': 0.0,
        'num_layers': 1}

seeds = 1
for seed in range(seeds):
    # Create experiment
    neptune.create_experiment('1D_20-step ahead predict_baseline_test', params = PARAMS, upload_source_files=['../LSTM_manager_1D.py', '1D_predictor.py'], tags=['single_run', 'no-extra-feature', '1D-prediction', '4-year', '20-step-ahead', '20-predictions', 'price','seed'+str(seed)])

    ############################  Single 20-step ahead prediction 6-D ##########################
    # FILEPATH = os.getenv('arg1')
    FILEPATH = './thesis_datasets/Dataset1/AAPL.csv'
    
    df = pd.read_csv(FILEPATH)	
    neptune.set_property('data', FILEPATH)
    
    getDataInfo(FILEPATH, seed)

    # feature 1
    data_name = 'Close'
    
    data = df.filter(items=[data_name])

    test_size = 20
    # The test size here is 20, this creates the split between what data is known and not known, like training and test.
    y_train, y_test = temporal_train_test_split(data[data_name], test_size=test_size)

    s = LSTM_manager_1D.LSTMHandler(seed = seed)
    # the test size in this case is 1, since we are only trying to predict 1 value, but 20 steps ahead. 
    s.create_train_test_data(data = data,
     data_name = data_name,
     test_size=test_size
     )

    s.create_trained_model(params=PARAMS)

    y_pred = s.make_predictions_from_model()

    y_test_to_predict = y_test[-test_size:]

    smape = smape_loss(y_test_to_predict, y_pred)
    neptune.log_metric('smape', smape)

    y_train = y_train
    # The rest of the input can be added later after scaling.
    fig, ax = plot_series(
        y_train,
        y_test, y_pred,
        labels=["y_train","y_test", "y_pred"]
        )

    neptune.log_image('univariate_plot', fig)
    ax.get_legend().remove()
    log_chart(name='univariate_plot', chart=fig)

    lossplot = s.plot_training_error()
    neptune.log_image('training_loss', lossplot)

    neptune.stop()