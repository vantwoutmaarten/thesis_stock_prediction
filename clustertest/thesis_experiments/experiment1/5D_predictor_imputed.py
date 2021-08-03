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

import thesis_experiments.experiment1.LSTM_manager_5D_imputed as LSTM_manager_5D_imputed


import optuna

import neptune
from neptunecontrib.api import log_chart
import os

neptune.init(project_qualified_name='mavantwout/Experiment-1',
             api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiODBkNzdjMDUtYmYxZi00ODFjLWExN2MtNjk3Y2MwZDE5N2U2In0=',
             )
def getDataInfo(datafilename, seed):
    split_on_data = datafilename.partition('/Dataset2/')
    split_for_missing =  split_on_data[2].partition('_missing')
    company = split_for_missing[0]
    missingness = split_for_missing[2].split('.')[0]

    imputation = '5_imputations_combined'

    print(company)
    print(imputation)

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
    neptune.create_experiment('5D_20-step ahead predict_exp1', params = PARAMS, upload_source_files=['./thesis_experiments/experiment1/LSTM_manager_5D_imputed.py', './thesis_experiments/experiment1/5D_predictor_imputed.py'], tags=['single_run', 'no-extra-feature', '5D-prediction', '4-year', '20-step-ahead', '20-predictions', 'price_imputed','seed'+str(seed)])

    ############################  Single 20-step ahead prediction 6-D ##########################
    FILEPATH = os.getenv('arg1')
    # FILEPATH = './thesis_datasets/Dataset2/AAPL_missing985.csv'
    
    df = pd.read_csv(FILEPATH)	
    neptune.set_property('data', FILEPATH)
    
    # imputer = os.getenv('arg2')
    getDataInfo(FILEPATH, seed)

    # output
    data_name = 'Close'
    # feature 1
    close_forwardfill = 'Close_missing98.5_forwardfill'
    # feature 2
    close_globalmean = 'Close_missing98.5_globalmean'
    # feature 3
    close_meanlast260 = 'Close_missing98.5_meanlast260'
    # feature 4
    close_linearfit260 = 'Close_missing98.5_linearfit260'
    # feature 5
    close_cubicfit260= 'Close_missing98.5_cubicfit260'


    data = df.filter(items=[data_name, close_forwardfill, close_globalmean, close_meanlast260, close_linearfit260, close_cubicfit260])

    test_size = 20
    # The test size here is 20, this creates the split between what data is known and not known, like training and test.
    y_train_out, y_test_out = temporal_train_test_split(data[data_name], test_size=test_size)
    train_forwardfill_in, _ = temporal_train_test_split(data[close_forwardfill], test_size=test_size)
    train_globalmean_in, _ = temporal_train_test_split(data[close_globalmean], test_size=test_size)
    train_meanlast260_in, _ = temporal_train_test_split(data[close_meanlast260], test_size=test_size)
    train_linearfit260in, _  = temporal_train_test_split(data[close_linearfit260], test_size=test_size)
    train_cubicfit260_in, _ = temporal_train_test_split(data[close_cubicfit260], test_size=test_size)

    s = LSTM_manager_5D_imputed.LSTMHandler(seed = seed)
    # the test size in this case is 1, since we are only trying to predict 1 value, but 20 steps ahead. 
    s.create_train_test_data(data = data,
     data_name = data_name,
     test_size=test_size
     )

    s.create_trained_model(params=PARAMS)

    y_pred = s.make_predictions_from_model()

    y_test_to_predict = y_test_out[-test_size:]

    smape = smape_loss(y_test_to_predict, y_pred)
    neptune.log_metric('smape', smape)

    # The rest of the input can be added later after scaling.
    fig, ax = plot_series(
        train_forwardfill_in,
        train_globalmean_in,
        train_meanlast260_in,
        train_linearfit260in,
        train_cubicfit260_in,
        y_train_out,
        y_test_out, y_pred,
        labels=["train_forwardfill_in","train_globalmean_in","train_meanlast260_in","train_linearfit260in","train_cubicfit260_in","y_train_out","y_test", "y_pred"]
        )

    ax.set_ylabel('Close')
    neptune.log_image('univariate_plot', fig)
    ax.get_legend().remove()
    log_chart(name='univariate_plot', chart=fig)

    lossplot = s.plot_training_error()
    neptune.log_image('training_loss', lossplot)

    neptune.stop()