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

# Create experiment
neptune.create_experiment('paramstest_1D', params = PARAMS, upload_source_files=['./LSTM_manager_2D.py', './LSTM_manager.py', './runner.py'], tags=['single_run', '1D-prediction', '4-year'])

############################  Single predictor  ########################## 
FILEPATH = os.getenv('arg1')
df = pd.read_csv(FILEPATH)
neptune.set_property('data', FILEPATH)

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