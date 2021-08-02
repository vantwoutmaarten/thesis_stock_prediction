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

import LSTM_manager_6D_20ahead_20pred

import optuna

import neptune
from neptunecontrib.api import log_chart
import os

neptune.init(project_qualified_name='mavantwout/excl-imputation-prices',
             api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiODBkNzdjMDUtYmYxZi00ODFjLWExN2MtNjk3Y2MwZDE5N2U2In0=',
             )
def getDataInfo(datafilename, seed):
    split_on_data = datafilename.partition('/imputed_data/')
    split_for_missing =  split_on_data[2].partition('/missing')
    company = split_for_missing[0]
    missingness = split_for_missing[2][:2]

    imputation = '5_imputations_combined'

    neptune.log_text('company', str(company))
    neptune.log_text('missingness', str(missingness))
    neptune.log_text('imputation', str(imputation))
    neptune.log_text('seed', str(seed))

PARAMS = {'epochs': 80,
        'lr':  0.00029177560092619997,
        'hls' : 28,
        'train_window': 20, 
        'opt' : 'RMSprop',
        'loss' : 'MSELoss',
        'dropout': 0.0,
        'num_layers': 1}

seeds = 5
for seed in range(seeds):
    # Create experiment
    neptune.create_experiment('6D_20-step ahead predict_returns', params = PARAMS, upload_source_files=['../timeseries_pytorch_simpleLSTM/LSTM_manager_6D_20ahead.py', '6D_predictor_20ahead_20pred.py'], tags=['single_run', '6D-prediction', '4-year', '20-step-ahead', '20-predictions', 'shifted30', 'minmax-11','returns','excluding_imputedvalues','seed'+str(seed)])

    ############################  Single 20-step ahead prediction 6-D ########################## 
    FILEPATH = os.getenv('arg1')
    getDataInfo(FILEPATH, seed)
    df = pd.read_csv(FILEPATH)	
    neptune.set_property('data', FILEPATH)

    data_name = 'Close'
    lagged_imputation_forwardfill ='Close_ahead30_missing90_forwardfill' 
    lagged_imputation_globalmean ='Close_ahead30_missing90_globalmean' 
    lagged_imputation_meanlast30 ='Close_ahead30_missing90_meanlast30' 
    lagged_imputation_linearfit30 ='Close_ahead30_missing90_linearfit30' 
    lagged_imputation_cubicfit30 ='Close_ahead30_missing90_cubicfit30' 
    
    data = df.filter(items=[data_name, lagged_imputation_forwardfill, lagged_imputation_globalmean, lagged_imputation_meanlast30, lagged_imputation_linearfit30, lagged_imputation_cubicfit30])


    test_size = 20
    # The test size here is 20, this creates the split between what data is known and not known, like training and test.
    y_train, y_test = temporal_train_test_split(data[data_name], test_size=test_size)

    y_train_lagged_forwardfill , y_test_lagged_forwardfill  = temporal_train_test_split(data[lagged_imputation_forwardfill], test_size=test_size)
    y_train_lagged_globalmean, y_test_lagged_globalmean = temporal_train_test_split(data[lagged_imputation_globalmean], test_size=test_size)
    y_train_lagged_meanlast30, y_test_lagged_meanlast30 = temporal_train_test_split(data[lagged_imputation_meanlast30], test_size=test_size)
    y_train_lagged_linearfit30, y_test_lagged_linearfit30 = temporal_train_test_split(data[lagged_imputation_linearfit30], test_size=test_size)
    y_train_lagged_cubicfit30, y_test_lagged_cubicfit30 = temporal_train_test_split(data[lagged_imputation_cubicfit30], test_size=test_size)




    s = LSTM_manager_6D_20ahead_20pred.LSTMHandler(seed = seed)
    # the test size in this case is 1, since we are only trying to predict 1 value, but 20 steps ahead. 
    s.create_train_test_data(data = data,
     data_name = data_name,
     train_lagged_forwardfill=y_train_lagged_forwardfill,
     train_lagged_globalmean=y_train_lagged_globalmean,
     train_lagged_meanlast30=y_train_lagged_meanlast30,
     train_lagged_linearfit30=y_train_lagged_linearfit30,
     train_lagged_cubicfit30=y_train_lagged_cubicfit30,
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



############## OPTIMIZER ##################
# # Create experiment
# neptune.create_experiment('2D_20-step ahead prediction_seed1_optimization_test', upload_source_files=['../timeseries_pytorch_simpleLSTM/LSTM_manager_2D_20ahead.py', '2D_predictor_20ahead.py'], tags=['optimization', 'single_run', '2D-prediction', '4-year', '20-step-ahead', '6shifted'])

#  ############################  Single 20-step ahead prediction 2-D ########################## 
# df = pd.read_csv("./synthetic_data/sinus_scenarios/2D_noisy_sin_period126_year4_ahead6_seed10.csv")

# data_name = 'noisy_sin'
# lagged_data_name = 'noisy_sin_lag6'
# data = df.filter(items=[data_name, lagged_data_name])

# # The test size here is 20, this creates the split between what data is known and not known, like training and test.
# y_train, y_test = temporal_train_test_split(data[data_name], test_size=20)
# y_train_lagged, y_test_lagged = temporal_train_test_split(data[lagged_data_name], test_size=20)

# s = LSTM_manager_2D_20ahead_20pred.LSTMHandler()
# # the test size in this case is 1, since we are only trying to predict 1 value, but 20 steps ahead. 
# s.create_train_test_data(data = data, data_name = data_name, lagged_data_name=lagged_data_name, test_size=20)


# s.optimize()

# neptune.stop()
# %%

# %%
