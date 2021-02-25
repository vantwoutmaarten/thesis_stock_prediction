# %%
from warnings import simplefilter

import numpy as np
import pandas as pd
from sktime.datasets import load_airline
from sktime.forecasting.arima import ARIMA, AutoARIMA
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import (
    EnsembleForecaster,
    ReducedRegressionForecaster,
    TransformedTargetForecaster,
)
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.model_selection import (
    ForecastingGridSearchCV,
    SlidingWindowSplitter,
    temporal_train_test_split,
)
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.performance_metrics.forecasting import sMAPE, smape_loss
from sktime.transformations.series.detrend import Deseasonalizer, Detrender
from sktime.utils.plotting import plot_series

from timeseries_pytorch_simpleLSTM import testmod
from timeseries_pytorch_simpleLSTM import LSTM_manager

simplefilter("ignore", FutureWarning)

%matplotlib inline

# %%
############################## Solver ##############################
###################### Type series: Upward brownian, missing values sol: no missing values comparison ############################
# df = pd.read_csv("./synthetic_data/brownian_scenarios/upward_mu_021_sig_065.csv")
# df = df['stockprice']
# df = pd.read_csv("synthetic_data/sinus_scenarios/noisy_sin_period20_missing20.csv")
# df = df['noisy_sin']
print("experiment testing the autoarima function and checking if sin+brownian motion can be predicted with autoARIMA, -> only possible for short sequences")
df = pd.read_csv("synthetic_data/sinus_scenarios/small_sin_period13_missing20.csv")
df = df['sinus']

# %%

y = df
# fig, ax = plot_series(y)
# ax.set(xlabel="days", ylabel="synthethic stock")
print(type(y))
# Split the airline data into train and test split.
y_train, y_test = temporal_train_test_split(y, test_size=50)
# plot_series(y_train, y_test, labels=["y_train", "y_test"])
print("the shape of the training and test is: ", y_train.shape[0], y_test.shape[0])

# Define the x values for the prediction. 
fh = np.arange(len(y_test)) + 1

# using the naive forecaster
print("naive forecaster")
forecaster = NaiveForecaster(strategy="last")
forecaster.fit(y_train)
y_pred = forecaster.predict(fh)
# plot_series(y_train, y_test, labels=["y_train", "y_test"])
plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
smape_naive = smape_loss(y_pred, y_test)
# %%
# using the linear forecaster
print("linear forecaster")
forecaster = PolynomialTrendForecaster(degree=1)
forecaster.fit(y_train)
y_pred = forecaster.predict(fh)
plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
smape_linear = smape_loss(y_pred, y_test)
#%%
# plot the detrender liner detrending
print("linear detrender")
forecaster = PolynomialTrendForecaster(degree=1)
transformer = Detrender(forecaster=forecaster)
yt = transformer.fit_transform(y_train)
# internally, the Detrender uses the in-sample predictions of the PolynomialTrendForecaster
forecaster = PolynomialTrendForecaster(degree=1)
fh_ins = -np.arange(len(y_train))  # in-sample forecasting horizon
fh_ins
y_pred = forecaster.fit(y_train).predict(fh=fh_ins)
plot_series(y_train, y_pred, yt, labels=["y_train", "fitted linear trend", "residuals"])
#%%
# plot the atuoarima solution
print("autoarima solution for short sequence")
forecaster = AutoARIMA(sp = 13, suppress_warnings=False)
y_pred = forecaster.fit(y_train).predict(fh)
plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
smape_ARIMA = smape_loss(y_test, y_pred)

print("For the upward stock no missing values,\n the SMAPE of the Naive forecaster is: ", "{:.3f}".format(smape_naive), "\n the linear forecaster is: ", "{:.3f}".format(smape_linear), "\n AutoArima forecaster is: ", "{:.3f}".format(smape_ARIMA))
# %%
###################### Type series: Upward brownian, missing values sol: fill with interpolation  ############################
###################### Type series: Upward brownian, missing values sol: fill with previous values ############################
# df = pd.read_csv("./synthetic_data/brownian_scenarios/upward_mu_021_sig_065.csv")
# data = df['stockprice_missing_closed_days']
# y = data
# fig, ax = plot_series(y)
# ax.set(xlabel="days", ylabel="synthethic data")
print("captures simple sin with ar")
forecaster = ARIMA(order = (12, 0, 0))
y_pred = forecaster.fit(y_train).predict(fh)
plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
smape_ARIMA = smape_loss(y_test, y_pred)
# %%
df = pd.read_csv("synthetic_data/sinus_scenarios/sin_period63_missing20.csv")
df = df['sinus']

y = df
# fig, ax = plot_series(y)
# ax.set(xlabel="days", ylabel="synthethic stock")

# Split the airline data into train and test split.
y_train, y_test = temporal_train_test_split(y, test_size=365)
# plot_series(y_train, y_test, labels=["y_train", "y_test"])
print("the shape of the training and test is: ", y_train.shape[0], y_test.shape[0])

# Define the x values for the prediction. 
fh = np.arange(len(y_test)) + 1

forecaster = ARIMA(order = (12, 0, 0))
y_pred = forecaster.fit(y_train).predict(fh)
plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
smape_ARIMA = smape_loss(y_test, y_pred)
# %%
df = pd.read_csv("synthetic_data/sinus_scenarios/sin_period251_missing20.csv")
df = df['sinus']

y = df
# fig, ax = plot_series(y)
# ax.set(xlabel="days", ylabel="synthethic stock")

# Split the airline data into train and test split.
y_train, y_test = temporal_train_test_split(y, test_size=365)
# plot_series(y_train, y_test, labels=["y_train", "y_test"])
print("the shape of the training and test is: ", y_train.shape[0], y_test.shape[0])

# Define the x values for the prediction. 
fh = np.arange(len(y_test)) + 1

forecaster = ARIMA(order = (12, 0, 0))
y_pred = forecaster.fit(y_train).predict(fh)
plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
smape_ARIMA = smape_loss(y_test, y_pred)
# %%
print("testing the AR models with different lags on sines with different periods + brownian noise")
df = pd.read_csv("synthetic_data/sinus_scenarios/noisy_sin_period126_missing20.csv")
df = df['noisy_sin']

y = df
# fig, ax = plot_series(y)
# ax.set(xlabel="days", ylabel="synthethic stock")

# Split the airline data into train and test split.
y_train, y_test = temporal_train_test_split(y, test_size=365)
# plot_series(y_train, y_test, labels=["y_train", "y_test"])
print("the shape of the training and test is: ", y_train.shape[0], y_test.shape[0])

# Define the x values for the prediction. 
fh = np.arange(len(y_test)) + 1

forecaster = ARIMA(order = (12, 0, 0))
y_pred = forecaster.fit(y_train).predict(fh)
plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
smape_ARIMA = smape_loss(y_test, y_pred)
smape_ARIMA
print("lag 12")
# %%
df = pd.read_csv("synthetic_data/sinus_scenarios/noisy_sin_period126_missing20.csv")
df = df['noisy_sin']

y = df
# fig, ax = plot_series(y)
# ax.set(xlabel="days", ylabel="synthethic stock")

# Split the airline data into train and test split.
y_train, y_test = temporal_train_test_split(y, test_size=365)
# plot_series(y_train, y_test, labels=["y_train", "y_test"])
print("the shape of the training and test is: ", y_train.shape[0], y_test.shape[0])

# Define the x values for the prediction. 
fh = np.arange(len(y_test)) + 1

forecaster = ARIMA(order = (20, 0, 0))
y_pred = forecaster.fit(y_train).predict(fh)
plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
smape_ARIMA = smape_loss(y_test, y_pred)
smape_ARIMA
print("lag 20")
# %%
df = pd.read_csv("synthetic_data/sinus_scenarios/noisy_sin_period126_missing20.csv")
df = df['noisy_sin']

y = df
# fig, ax = plot_series(y)
# ax.set(xlabel="days", ylabel="synthethic stock")

# Split the airline data into train and test split.
y_train, y_test = temporal_train_test_split(y, test_size=365)
# plot_series(y_train, y_test, labels=["y_train", "y_test"])
print("the shape of the training and test is: ", y_train.shape[0], y_test.shape[0])

# Define the x values for the prediction. 
fh = np.arange(len(y_test)) + 1

forecaster = ARIMA(order = (30, 0, 0))
y_pred = forecaster.fit(y_train).predict(fh)
plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
smape_ARIMA = smape_loss(y_test, y_pred)
smape_ARIMA
print("lag 30")
# %%
df = pd.read_csv("synthetic_data/sinus_scenarios/noisy_sin_period251_missing20.csv")
df = df['noisy_sin']

y = df
# fig, ax = plot_series(y)
# ax.set(xlabel="days", ylabel="synthethic stock")

# Split the airline data into train and test split.
y_train, y_test = temporal_train_test_split(y, test_size=365)
# plot_series(y_train, y_test, labels=["y_train", "y_test"])
print("the shape of the training and test is: ", y_train.shape[0], y_test.shape[0])

# Define the x values for the prediction. 
fh = np.arange(len(y_test)) + 1

forecaster = ARIMA(order = (12, 0, 0))
y_pred = forecaster.fit(y_train).predict(fh)
plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
smape_ARIMA = smape_loss(y_test, y_pred)
smape_ARIMA
print("lag 12")
# %%
df = pd.read_csv("synthetic_data/sinus_scenarios/noisy_sin_period251_missing20.csv")
df = df['noisy_sin']

y = df
# fig, ax = plot_series(y)
# ax.set(xlabel="days", ylabel="synthethic stock")

# Split the airline data into train and test split.
y_train, y_test = temporal_train_test_split(y, test_size=365)
# plot_series(y_train, y_test, labels=["y_train", "y_test"])
print("the shape of the training and test is: ", y_train.shape[0], y_test.shape[0])

# Define the x values for the prediction. 
fh = np.arange(len(y_test)) + 1

forecaster = ARIMA(order = (20, 0, 0))
y_pred = forecaster.fit(y_train).predict(fh)
plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
smape_ARIMA = smape_loss(y_test, y_pred)
smape_ARIMA
print("lag 20")
# %%
print("experiment to check if AR can predict sin + little noise stochastic")
df = pd.read_csv("synthetic_data/sinus_scenarios/stochastic015_sin_period31_missing20.csv")
df = df['noisy_sin']

y = df
# fig, ax = plot_series(y)
# ax.set(xlabel="days", ylabel="synthethic stock")

# Split the airline data into train and test split.
y_train, y_test = temporal_train_test_split(y, test_size=365)
# plot_series(y_train, y_test, labels=["y_train", "y_test"])
print("the shape of the training and test is: ", y_train.shape[0], y_test.shape[0])

# Define the x values for the prediction. 
fh = np.arange(len(y_test)) + 1

forecaster = ARIMA(order = (31, 0, 0))
y_pred = forecaster.fit(y_train).predict(fh)
plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
smape_ARIMA = smape_loss(y_test, y_pred)
smape_ARIMA
# %%
print("experiment to check if AR can predict sin + double amount of noise stochastic")
df = pd.read_csv("synthetic_data/sinus_scenarios/stochastic03_sin_period31_missing20.csv")
df = df['noisy_sin']

y = df
# fig, ax = plot_series(y)
# ax.set(xlabel="days", ylabel="synthethic stock")

# Split the airline data into train and test split.
y_train, y_test = temporal_train_test_split(y, test_size=365)
# plot_series(y_train, y_test, labels=["y_train", "y_test"])
print("the shape of the training and test is: ", y_train.shape[0], y_test.shape[0])

# Define the x values for the prediction. 
fh = np.arange(len(y_test)) + 1

forecaster = ARIMA(order = (31, 0, 0))
y_pred = forecaster.fit(y_train).predict(fh)
plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
smape_ARIMA = smape_loss(y_test, y_pred)
smape_ARIMA
# %%
print("experiment to check if AR can predict LONGER sin + double amount of noise stochastic")
df = pd.read_csv("synthetic_data/sinus_scenarios/stochastic03_sin_period63_missing20.csv")
df = df['noisy_sin']

y = df
# fig, ax = plot_series(y)
# ax.set(xlabel="days", ylabel="synthethic stock")

# Split the airline data into train and test split.
y_train, y_test = temporal_train_test_split(y, test_size=365)
# plot_series(y_train, y_test, labels=["y_train", "y_test"])
print("the shape of the training and test is: ", y_train.shape[0], y_test.shape[0])

# Define the x values for the prediction. 
fh = np.arange(len(y_test)) + 1

forecaster = ARIMA(order = (31, 0, 0))
y_pred = forecaster.fit(y_train).predict(fh)
plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
smape_ARIMA = smape_loss(y_test, y_pred)
smape_ARIMA
# %%
# experiment to see if ARIMA can capture the sin + brownian noise when the correct values for p(ACF) autoregressive d and q(with PACF) for moving average are chosen
from pandas.plotting import autocorrelation_plot

df = pd.read_csv("synthetic_data/sinus_scenarios/noisy_sin_period126_missing20.csv")
df = df['noisy_sin']
y = df
y_train, y_test = temporal_train_test_split(y, test_size=365)
print("the shape of the training and test is: ", y_train.shape[0], y_test.shape[0])
fh = np.arange(len(y_test)) + 1

autocorrelation_plot(y_train)
#many lags are above critical boundry so this does not matter

# %%
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(y_train, lags=20)
#it seems that 11 is very important so q = 11
# %%
forecaster = ARIMA(order = (50, 1, 11))
y_pred = forecaster.fit(y_train).predict(fh)
plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
smape_ARIMA = smape_loss(y_test, y_pred)
smape_ARIMA
# %%
forecaster = ARIMA(order = (60, 1, 11))
y_pred = forecaster.fit(y_train).predict(fh)
plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
smape_ARIMA = smape_loss(y_test, y_pred)
smape_ARIMA
# %%
############################## lstm work ###########################
############# Experiment manually optimizing hyperparameters trying to predict the sin + brownian motion

# %%
df = pd.read_csv("./synthetic_data/sinus_scenarios/noisy_sin_period126_missing20.csv")
data_name = 'noisy_sin'
data = df.filter([data_name])


y = data[data_name]
y_train, y_test = temporal_train_test_split(y, test_size=365)


s = LSTM_manager.LSTMHandler()
s.create_train_test_data(data = data, data_name = data_name)

# %%

y_pred = s.make_predictions_from_model(modelpath="timeseries_pytorch_simpleLSTM/noisy_sin_period126_epochs15_window252_HN40_lr00004_try2.pt")

plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])

# %%
y_pred = s.make_predictions_from_model(modelpath="timeseries_pytorch_simpleLSTM/noisy_sin_period126_epochs15_window252_HN40_lr00004_try2.pt")

plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])

# %%
y_pred = s.make_predictions_from_model(modelpath="timeseries_pytorch_simpleLSTM/noisy_sin_period126_epochs15_window252_HN40_lr00004.pt")

plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])

# %%
s.create_trained_model(modelpath="timeseries_pytorch_simpleLSTM/noisy_sin_period126_epochs6_window252_HN40_lr00004.pt", epochs=6)


y_pred = s.make_predictions_from_model(modelpath="timeseries_pytorch_simpleLSTM/noisy_sin_period126_epochs6_window252_HN40_lr00004.pt")

plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])

s.plot_training_error()