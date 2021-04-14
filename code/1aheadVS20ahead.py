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

# Create experiment
neptune.create_experiment('1-step ahead prediction_seed1', params = PARAMS, upload_source_files=['../timeseries_pytorch_simpleLSTM/LSTM_manager1vs20.py', '../timeseries_pytorch_simpleLSTM/LSTM_manager.py', '1aheadVS20ahead.py'], tags=['single_run', '1D-prediction', '7-year', '1-step-ahead'])

 ############################  Single 1-step ahead prediction ########################## 
df = pd.read_csv("./synthetic_data/sin_brownian_for_optimization/1D_noisy_sin_period126_year7_seed1.csv")

data_name = 'noisy_sin'
data = df.filter([data_name])

y = data[data_name]
y_train, y_test = temporal_train_test_split(y, test_size=1)
s = LSTM_manager.LSTMHandler()

s.create_train_test_data(data = data, data_name = data_name, test_size=1)

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