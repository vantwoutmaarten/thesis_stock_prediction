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

neptune.init(project_qualified_name='mavantwout/thesis-optimization',
             api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiODBkNzdjMDUtYmYxZi00ODFjLWExN2MtNjk3Y2MwZDE5N2U2In0=',
             )
def getDataInfo(datafilename):
    split_on_data = datafilename.partition('/Dataset4/')
    split_for_missing =  split_on_data[2].partition('_Price')
    company = split_for_missing[0]

    imputation = '2_imputations_combined'

    print(company)
    print(imputation)

    neptune.log_text('company', str(company))
    neptune.log_text('imputation', str(imputation))

# Create experiment
neptune.create_experiment('12D_20-step ahead predict_exp5_test-optimization', upload_source_files=['../LSTM_manager_12D.py', '12D_predictor_20ahead_tuner.py'], tags=['optimization', '12D-prediction', '4-year', '20-step-ahead', '20-predictions','quarterly'])

############################  Single 20-step ahead prediction 6-D ##########################
# FILEPATH = os.getenv('arg1')
FILEPATH = './thesis_datasets/Dataset4/AAPL_Price_and_Quarterly.csv'
getDataInfo(FILEPATH)
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

s = LSTM_manager_12D.LSTMHandler()
# the test size in this case is 1, since we are only trying to predict 1 value, but 20 steps ahead. 
s.create_train_test_data(data = data,
    data_name = data_name,
    test_size=test_size
    )

s.optimize()

neptune.stop()
