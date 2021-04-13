from sktime.forecasting.model_selection import (
    ForecastingGridSearchCV,
    SlidingWindowSplitter,
    temporal_train_test_split,
)
import numpy as np
import pandas as pd
from sktime.utils.plotting import plot_series
from sktime.performance_metrics.forecasting import sMAPE, smape_loss, mape_loss

import LSTM_manager

import LSTM_manager_2D
import optuna
import os

################################################# neptune test 1 #################################
import neptune

neptune.init(project_qualified_name='mavantwout/hyperparam',
             api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiODBkNzdjMDUtYmYxZi00ODFjLWExN2MtNjk3Y2MwZDE5N2U2In0=',
             )

# Create experiment
neptune.create_experiment('paramtuner_1D', upload_source_files=['./LSTM_manager_2D.py', './LSTM_manager.py', './hyperparameter2D.py'], tags=['single_run', '2D-prediction', '7-year', 'hyperparameter-test'])

FILEPATH = os.getenv('arg1')
df = pd.read_csv(FILEPATH)
neptune.set_property('data', FILEPATH)

data_name = 'noisy_sin'
data = df.filter([data_name])

y = data[data_name]
y_train, y_test = temporal_train_test_split(y, test_size=365)
s = LSTM_manager.LSTMHandler()

s.create_train_test_data(data = data, data_name = data_name, test_size=365)

s.optimize()

neptune.stop()
