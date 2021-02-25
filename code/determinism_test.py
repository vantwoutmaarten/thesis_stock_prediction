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


# %%
df = pd.read_csv("./synthetic_data/brownian_scenarios/downward_mu_021_sig_065.csv")
data_name = 'stockprice'
data = df.filter([data_name])

y = data[data_name]
y_train, y_test = temporal_train_test_split(y, test_size=365)
s = LSTM_manager.LSTMHandler()
s.create_train_test_data(data = data, data_name = data_name)
 # %%
s.create_trained_model(modelpath="./testmodel4.pt", epochs=10)

#%%
y_pred = s.make_predictions_from_model(modelpath="./testmodel4.pt")

plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])

s.plot_training_error()
# %%
