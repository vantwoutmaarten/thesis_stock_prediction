from sktime.forecasting.model_selection import (
    ForecastingGridSearchCV,
    SlidingWindowSplitter,
    temporal_train_test_split,
)
import numpy as np
import pandas as pd
from sktime.utils.plotting import plot_series
from sktime.performance_metrics.forecasting import sMAPE, smape_loss, mape_loss

from timeseries_pytorch_simpleLSTM import LSTM_manager

from timeseries_pytorch_simpleLSTM import LSTM_manager_2D
import optuna

#%%
 ############################  TEST 1  ########################## 

# df = pd.read_csv("./synthetic_data/sinus_scenarios/noisy_sin_period126_missing20.csv")
# # df = pd.read_csv("./synthetic_data/sinus_scenarios/noisy_sin_period126_missing20_year_10.csv")
# # df = pd.read_csv("./synthetic_data/sinus_scenarios/noisy_sin_period63_missing20_year4.csv")
# # df = pd.read_csv("./synthetic_data/sinus_scenarios/noisy_sin_period63_missing20_year10.csv")

# data_name = 'noisy_sin'
# data = df.filter([data_name])

# y = data[data_name]
# y_train, y_test = temporal_train_test_split(y, test_size=365)
# s = LSTM_manager.LSTMHandler()

# s.create_train_test_data(data = data, data_name = data_name, test_size=365)

# s.create_trained_model(epochs=1, lr = 0.0003152924226911524, hidden_layer_size = 11, train_window=255, optimizer_name='RMSprop', loss_name = 'MSELoss', dropout=0.0, num_layers=1)
# y_pred = s.make_predictions_from_model()
# plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
# s.plot_training_error()

# s.optimize()

# %%
################################################## neptune test 1 #################################
# import neptune

# neptune.init(project_qualified_name='mavantwout/sandbox',
#              api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiODBkNzdjMDUtYmYxZi00ODFjLWExN2MtNjk3Y2MwZDE5N2U2In0=',
#              )

# # Create experiment
# neptune.create_experiment('neptune-optuna-test', upload_source_files=['../timeseries_pytorch_simpleLSTM/LSTM_manager_2D.py', '../timeseries_pytorch_simpleLSTM/LSTM_manager.py', 'hyperparameter_tuner.py'])

# df = pd.read_csv("./synthetic_data/sinus_scenarios/2D_noisy_sin_period126_year4_lag6_seed10.csv")

# data_name = 'noisy_sin'
# data = df.filter([data_name])

# y = data[data_name]
# y_train, y_test = temporal_train_test_split(y, test_size=365)
# s = LSTM_manager.LSTMHandler()

# s.create_train_test_data(data = data, data_name = data_name, test_size=365)

# s.optimize()

# neptune.stop()
# %%
############################## test 2 ############################
############ 2D optimization
import neptune

neptune.init(project_qualified_name='mavantwout/sandbox',
             api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiODBkNzdjMDUtYmYxZi00ODFjLWExN2MtNjk3Y2MwZDE5N2U2In0=',
             )

neptune.create_experiment('hyperparamater_test_2D')

df = pd.read_csv("./synthetic_data/sinus_scenarios/2D_noisy_sin_period126_year4_lag6_seed10.csv")

data_name = 'noisy_sin'
lagged_data_name = 'noisy_sin_lag6'
data = df.filter(items=[data_name, lagged_data_name])

y_train, y_test = temporal_train_test_split(data[data_name], test_size=365)
y_train_lagged, y_test_lagged = temporal_train_test_split(data[lagged_data_name], test_size=365)

s = LSTM_manager_2D.LSTMHandler()

s.create_train_test_data(data = data, data_name = data_name, lagged_data_name=lagged_data_name, test_size=365)

s.optimize()

neptune.stop()
