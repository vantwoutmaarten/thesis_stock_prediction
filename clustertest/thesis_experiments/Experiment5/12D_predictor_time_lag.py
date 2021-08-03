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

import thesis_experiments.experiment5.LSTM_manager_12D as LSTM_manager_12D


import optuna

import neptune
from neptunecontrib.api import log_chart
import os

neptune.init(project_qualified_name='mavantwout/Experiment-5',
             api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiODBkNzdjMDUtYmYxZi00ODFjLWExN2MtNjk3Y2MwZDE5N2U2In0=',
             )
def getDataInfo(datafilename, seed):
    split_on_data = datafilename.partition('/Dataset4/')
    split_for_missing =  split_on_data[2].partition('_Price')
    company = split_for_missing[0]

    imputation = '2_imputations_combined'

    print(company)
    print(imputation)

    neptune.log_text('company', str(company))
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
    neptune.create_experiment('12D_20-step ahead predict_exp5_timelag', params = PARAMS, upload_source_files=['./thesis_experiments/experiment5/LSTM_manager_12D.py', './thesis_experiments/experiment5/12D_predictor_.py'], tags=['single_run', 'time-lag', '12D-prediction', '4-year', '20-step-ahead', '20-predictions', 'quarterly','seed'+str(seed)])

    ############################  Single 20-step ahead prediction 6-D ##########################
    # FILEPATH = os.getenv('arg1')
    FILEPATH = './thesis_datasets/Dataset4/AAPL_Price_and_Quarterly.csv'
    getDataInfo(FILEPATH, seed)
    df = pd.read_csv(FILEPATH)	
    neptune.set_property('data', FILEPATH)

    # feature 1
    data_name = 'Close'
    # feature 2, 3
    EnterpriseValue_meanlast260 ='EnterpriseValue_meanlast260'
    EnterpriseValue_linearfit260 = 'EnterpriseValue_linearfit260'
    # feature 4, 5
    PeRatio_meanlast260 = 'PeRatio_meanlast260'
    PeRatio_linearfit260 = 'PeRatio_linearfit260'
    # feature 6, 7
    ForwardPeRatio_meanlast260 = 'ForwardPeRatio_meanlast260'
    ForwardPeRatio_linearfit260 = 'ForwardPeRatio_linearfit260'
    # feature 8, 9
    PegRatio_meanlast260 = 'PegRatio_meanlast260'
    PegRatio_linearfit260 = 'PegRatio_linearfit260'
    # feature 10, 11
    EnterprisesValueEBITDARatio_meanlast260 = 'EnterprisesValueEBITDARatio_meanlast260'
    EnterprisesValueEBITDARatio_linearfit260 = 'EnterprisesValueEBITDARatio_linearfit260'
    # feature 12
    time_lag = 'time_lag'
    
    data = df.filter(items=[data_name, EnterpriseValue_meanlast260, EnterpriseValue_linearfit260, PeRatio_meanlast260,
     PeRatio_linearfit260, ForwardPeRatio_meanlast260, ForwardPeRatio_linearfit260, PegRatio_meanlast260, PegRatio_linearfit260, EnterprisesValueEBITDARatio_meanlast260, EnterprisesValueEBITDARatio_linearfit260, time_lag])

    test_size = 20
    # The test size here is 20, this creates the split between what data is known and not known, like training and test.
    y_train, y_test = temporal_train_test_split(data[data_name], test_size=test_size)

    s = LSTM_manager_12D.LSTMHandler(seed = seed)
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

    # make the general scaler for all the columns and make the fitted scaler for the y_pred
    scaler = MinMaxScaler(feature_range=(-1, 1))
    fittedscaler = scaler.fit(data[data_name].values.reshape(-1,1))
    scaler = MinMaxScaler(feature_range=(-1, 1))

    data_scaled = scaler.fit_transform(data)
    data = pd.DataFrame(data_scaled, columns=data.columns)

    y_train, y_test = temporal_train_test_split(data[data_name], test_size=test_size)

    y_train_EV_meanlast260 , y_test_EV_meanlast260 = temporal_train_test_split(data[EnterpriseValue_meanlast260], test_size=test_size)
    y_train_EV_linearfit260 , y_test_EV_linearfit260= temporal_train_test_split(data[EnterpriseValue_linearfit260], test_size=test_size)

    y_train_PE_meanlast260, y_test_PE_meanlast260= temporal_train_test_split(data[PeRatio_meanlast260], test_size=test_size)
    y_train_PE_linearfit260, y_test_PE_linearfit260= temporal_train_test_split(data[PeRatio_linearfit260], test_size=test_size)

    y_train_Forward_PE_meanlast260, y_test_Forward_PE_meanlast260= temporal_train_test_split(data[ForwardPeRatio_meanlast260], test_size=test_size)
    y_train_Forward_PE_linearfit260, y_test_Forward_PE_linearfit260= temporal_train_test_split(data[ForwardPeRatio_linearfit260], test_size=test_size)

    y_train_PEG_meanlast260, y_test_PEG_meanlast260= temporal_train_test_split(data[PegRatio_meanlast260], test_size=test_size)
    y_train_PEG_linearfit260, y_test_PEG_linearfit260= temporal_train_test_split(data[PegRatio_linearfit260], test_size=test_size)

    y_train_EV_EBITDA_meanlast260, y_test_EV_EBITDA_meanlast260= temporal_train_test_split(data[EnterprisesValueEBITDARatio_meanlast260], test_size=test_size)
    y_train_EV_EBITDA_linearfit260, y_test_EV_EBITDA_linearfit260= temporal_train_test_split(data[EnterprisesValueEBITDARatio_linearfit260], test_size=test_size)

    y_train_time_lag, y_test_time_lag= temporal_train_test_split(data[time_lag], test_size=test_size)

  
    indexpred = list(y_pred.index)
    y_pred = fittedscaler.transform(y_pred.values.reshape(-1,1))
    y_pred = pd.Series(y_pred.reshape(-1))
    y_pred.index = indexpred

    fig2, ax = plot_series(
    y_train,
    y_train_EV_meanlast260,
    y_train_PE_meanlast260,
    y_train_Forward_PE_meanlast260,
    y_train_PEG_meanlast260,
    y_train_EV_EBITDA_meanlast260,
    y_test, y_pred,
    labels=["y_train","EV_meanlast260","PE_meanlast260","Forward_PE_meanlast260",
    "PEG_meanlast260","EV_EBITDA_meanlast260","y_test", "y_pred"]
    )

    neptune.log_image('univariate_plot_scaled1', fig2)
    ax.get_legend().remove()
    log_chart(name='univariate_plot_scaled1', chart=fig2)

    # fig3, ax = plot_series(
    # y_train,
    # y_train_EV_meanlast260,
    # y_train_EV_linearfit260,
    # y_train_PE_meanlast260,
    # y_train_PE_linearfit260,
    # y_train_Forward_PE_meanlast260,
    # y_train_Forward_PE_linearfit260,
    # y_train_PEG_meanlast260,
    # y_train_PEG_linearfit260,
    # y_train_EV_EBITDA_meanlast260,
    # y_train_EV_EBITDA_linearfit260,
    # y_train_time_lag,
    # y_test, y_pred,
    # labels=["y_train","EV_meanlast260","EV_linearfit260","PE_meanlast260", "linearfit260","Forward_PE_meanlast260","Forward_PE_linearfit260",
    # "PEG_meanlast260","PEG_linearfit260","EV_EBITDA_meanlast260","EV_EBITDA_linearfit260","time_lag", "y_test", "y_pred"]
    # )
    fig3, ax = plot_series(
    y_train,
    y_train_EV_linearfit260,
    y_train_PE_linearfit260,
    y_train_Forward_PE_linearfit260,
    y_train_PEG_linearfit260,
    y_train_EV_EBITDA_linearfit260,
    y_test, y_pred,
    labels=["y_train","EV_linearfit260", "PE_linearfit260","Forward_PE_linearfit260",
    "PEG_linearfit260","EV_EBITDA_linearfit260","y_test", "y_pred"]
    )

    neptune.log_image('univariate_plot_scaled2', fig3)

    lossplot = s.plot_training_error()
    neptune.log_image('training_loss', lossplot)
    
    ax.get_legend().remove()
    log_chart(name='univariate_plot_scaled2', chart=fig3)

    neptune.stop()