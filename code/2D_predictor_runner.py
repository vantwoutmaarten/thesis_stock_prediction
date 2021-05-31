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


from timeseries_pytorch_simpleLSTM import LSTM_manager_2D
from timeseries_pytorch_simpleLSTM import LSTM_manager

import optuna

import neptune
from neptunecontrib.api import log_chart
    
neptune.init(project_qualified_name='mavantwout/sandbox',
             api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiODBkNzdjMDUtYmYxZi00ODFjLWExN2MtNjk3Y2MwZDE5N2U2In0=',
             )

PARAMS = {'epochs': 15,
        'lr': 0.000315292,
        'hls' : 20,
        'train_window': 95, 
        'opt' : 'Adam',
        'loss' : 'MSELoss',
        'dropout': 0.0,
        'num_layers': 1}

# Create experiment
neptune.create_experiment('paramstest_1D', params = PARAMS, upload_source_files=['../timeseries_pytorch_simpleLSTM/LSTM_manager_2D.py', '../timeseries_pytorch_simpleLSTM/LSTM_manager.py', '2D_predictor_runner.py'])

#%%
 ############################  Single predictor  ########################## 
df = pd.read_csv("./synthetic_data/sinus_scenarios/2D_noisy_sin_period126_year4_lag6_seed10.csv")

data_name = 'noisy_sin'
data = df.filter([data_name])

y = data[data_name]
y_train, y_test = temporal_train_test_split(y, test_size=365)
s = LSTM_manager.LSTMHandler()

s.create_train_test_data(data = data, data_name = data_name, test_size=365)

s.create_trained_model(params=PARAMS)

y_pred_univariate = s.make_predictions_from_model()

smape = smape_loss(y_test, y_pred_univariate)
neptune.log_metric('smape', smape)



fig, ax = plot_series(y_train, y_test, y_pred_univariate, labels=["y_train", "y_test", "y_pred_univariate"])

neptune.log_image('univariate_plot', fig)

lossplot = s.plot_training_error()
neptune.log_image('training_loss', lossplot)

ax.get_legend().remove()
log_chart(name='univariate_plot', chart=fig)


neptune.stop()
 #%%
 ############################  TEST 2  ########################## 
neptune.create_experiment('paramstest_2D', params = PARAMS)
df = pd.read_csv("./synthetic_data/sinus_scenarios/2D_noisy_sin_period126_year4_lag6_seed10.csv")

data_name = 'noisy_sin'
lagged_data_name = 'noisy_sin_lag6'
data = df.filter(items=[data_name, lagged_data_name])

y_train, y_test = temporal_train_test_split(data[data_name], test_size=365)
y_train_lagged, y_test_lagged = temporal_train_test_split(data[lagged_data_name], test_size=365)

s = LSTM_manager_2D.LSTMHandler()

s.create_train_test_data(data = data, data_name = data_name, lagged_data_name=lagged_data_name, test_size=365)

s.create_trained_model(params=PARAMS)

y_pred, y_pred_lag = s.make_predictions_from_model()
# plot_series(y_train, y_train_lagged, y_test, y_test_lagged, y_pred, y_pred_lag , labels=["y_train","y_train_lagged", "y_test", "y_test_lagged", "y_pred", "y_pred_lag"])

fig, ax = plot_series(y_train, y_train_lagged, y_test, y_test_lagged, y_pred, y_pred_lag , labels=["y_train", "y_train_lagged", "y_test", "y_test_lagged", "y_pred", "y_pred_lag"])

neptune.log_image('2D_with_lagged_plot', fig)

lossplot = s.plot_training_error()
neptune.log_image('training_loss', lossplot)

ax.get_legend().remove()
log_chart(name='2D_with_lagged_plot', chart=fig)

smape_normal = smape_loss(y_test, y_pred)
neptune.log_metric('smape', smape_normal)

smape_lagged = smape_loss(y_test_lagged, y_pred_lag)
neptune.log_metric('smape_of_lagged_series', smape_lagged)

neptune.stop()
# %%

# plot_series(y_test, y_test_lagged, y_pred, y_pred_lag, y_pred_univariate, labels=["y_test", "y_test_lagged", "y_pred", "y_pred_lag", "y_pred_univariate"])

# s.plot_training_error()

# s.optimize()
