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

from timeseries_pytorch_simpleLSTM import LSTM_manager
import optuna

# %%
df = pd.read_csv("./synthetic_data/sinus_scenarios/noisy_sin_period126_missing20.csv")
data_name = 'noisy_sin'
data = df.filter([data_name])

y = data[data_name]
y_train, y_test = temporal_train_test_split(y, test_size=365)
s = LSTM_manager.LSTMHandler()

s.create_train_test_data(data = data, data_name = data_name, test_size=365)


# %%
s.optimize()

# %%
s.create_trained_model(epochs=14, lr = 0.008540, hidden_layer_size = 68, train_window=399, optimizer_name='Adam', loss_name = 'MSELoss', dropout=0.5, num_layers=2)

# %%
y_pred = s.make_predictions_from_model()

plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
# %%
s.plot_training_error()
# %%
