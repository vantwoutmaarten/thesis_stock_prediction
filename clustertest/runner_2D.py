from sktime.forecasting.model_selection import (
    ForecastingGridSearchCV,
    SlidingWindowSplitter,
    temporal_train_test_split,
)
import numpy as np
import pandas as pd
from sktime.utils.plotting import plot_series
from sktime.performance_metrics.forecasting import sMAPE, smape_loss


import LSTM_manager_2D
import LSTM_manager

import optuna

import neptune
from neptunecontrib.api import log_chart
import os

neptune.init(project_qualified_name='mavantwout/cluster',
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

neptune.create_experiment('paramstest_2D', params = PARAMS, upload_source_files=['./LSTM_manager_2D.py', './LSTM_manager.py', './runner.py'], tags=['single_run', '2D-prediction', '7-year'])

FILEPATH = os.getenv('arg1')
df = pd.read_csv(FILEPATH)
neptune.set_property('data', FILEPATH)

data_name = 'noisy_sin'
lagged_data_name = 'noisy_sin_lag30'
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

ax.get_legend().remove()
log_chart(name='2D_with_lagged_plot', chart=fig)

lossplot = s.plot_training_error()
neptune.log_image('training_loss', lossplot)

smape_normal = smape_loss(y_test, y_pred)
neptune.log_metric('smape', smape_normal)

smape_lagged = smape_loss(y_test_lagged, y_pred_lag)
neptune.log_metric('smape_of_lagged_series', smape_lagged)

neptune.stop()