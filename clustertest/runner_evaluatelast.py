from sktime.forecasting.model_selection import (
    ForecastingGridSearchCV,
    SlidingWindowSplitter,
    temporal_train_test_split,
)
import numpy as np
import pandas as pd
from sktime.utils.plotting import plot_series
from sktime.performance_metrics.forecasting import sMAPE, smape_loss, mape_loss


import LSTM_manager_2D
import LSTM_manager

import optuna

import neptune
from neptunecontrib.api import log_chart
import os

neptune.init(project_qualified_name='mavantwout/1vsKaheadPrediction',
             api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiODBkNzdjMDUtYmYxZi00ODFjLWExN2MtNjk3Y2MwZDE5N2U2In0=',
             )

PARAMS = {'epochs': 1,
        'lr': 0.0005,
        'hls' : 86,
        'train_window': 247, 
        'opt' : 'RMSprop',
        'loss' : 'MSELoss',
        'dropout': 0.0,
        'num_layers': 1}

# Create experiment
neptune.create_experiment('recursive_20-step ahead prediction_seed1', params = PARAMS, upload_source_files=['./LSTM_manager1vs20.py', './LSTM_manager.py', './1aheadVS20ahead.py'], tags=['single_run', '1D-prediction', '7-year', '20-step-ahead', 'direct prediction', 'single_prediction'])

############################  Single predictor  ########################## 
FILEPATH = os.getenv('arg1')
df = pd.read_csv(FILEPATH)
neptune.set_property('data', FILEPATH)

data_name = 'noisy_sin'
data = df.filter([data_name])

y = data[data_name]
y_train, y_test = temporal_train_test_split(y, test_size=20)
s = LSTM_manager.LSTMHandler()

s.create_train_test_data(data = data, data_name = data_name, test_size=20)

s.create_trained_model(params=PARAMS)

y_pred_univariate = s.make_predictions_from_model()


y_test_last = y_test[-1:]
y_pred_univariate_last = y_pred_univariate[-1:]

smape = smape_loss(y_test_last, y_pred_univariate_last)
neptune.log_metric('smape', smape)
mape = mape_loss(y_test, y_pred_univariate)
neptune.log_metric('mape', mape)



print("no plots yet")
fig, ax = plot_series(y_train, y_test, y_pred_univariate, labels=["y_train", "y_test", "y_pred_univariate"])
print("plotted series")

neptune.log_image('univariate_plot', fig)

ax.get_legend().remove()
log_chart(name='univariate_plot', chart=fig)

lossplot = s.plot_training_error()
print("plotted training error")
neptune.log_image('training_loss', lossplot)

neptune.stop()