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

simplefilter("ignore", FutureWarning)

# %matplotlib inline
# %%
############################ AIRLINE EXAMPLE ############################
#     y = load_airline()
#     fig, ax = plot_series(y)
#     ax.set(xlabel="Time", ylabel="Number of airline passengers")

#     # Split the airline data into train and test split.
#     y_train, y_test = temporal_train_test_split(y, test_size=36)
#     # plot_series(y_train, y_test, labels=["y_train", "y_test"])
#     print("the shape of the training and test is: ", y_train.shape[0], y_test.shape[0])

#     # Define the x values for the prediction. 
#     fh = np.arange(len(y_test)) + 1
#     fh

#     # using the naive forecaster of sktime
#     forecaster = NaiveForecaster(strategy="last")
#     forecaster.fit(y_train)
#     y_pred = forecaster.predict(fh)
#     # plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
#     smape_naive = smape_loss(y_pred, y_test)
#     # %%
#     # plot the detrender liner detrending
#     forecaster = PolynomialTrendForecaster(degree=1)
#     transformer = Detrender(forecaster=forecaster)
#     yt = transformer.fit_transform(y_train)

#     # internally, the Detrender uses the in-sample predictions
#     # of the PolynomialTrendForecaster
#     forecaster = PolynomialTrendForecaster(degree=1)
#     fh_ins = -np.arange(len(y_train))  # in-sample forecasting horizon
#     fh_ins
#     y_pred = forecaster.fit(y_train).predict(fh=fh_ins)

#     plot_series(y_train, y_pred, yt, labels=["y_train", "fitted linear trend", "residuals"]);
#     # %%
#     forecaster = AutoARIMA(sp=12, suppress_warnings=True)
#     forecaster.fit(y_train)
#     y_pred = forecaster.predict(fh)
#     plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
#     smape_ARIMA = smape_loss(y_test, y_pred)
# print("For the airlines, the SMAPE of the Naive forecaster is: ", smape_naive, " and the SMAPE of the AutoArima forecaster is: ", smape_ARIMA)

# %%
# AutoArima with optimizer grid, but the best ARIMA is chosen already, so the grid is not useful here.
    # param_grid = {"sp": [2, 5, 12]}
    # forecaster = AutoARIMA(sp=12, suppress_warnings=True)
    # #  we fit the forecaster on the initial window,
    # # and then use temporal cross-validation to find the optimal parameter
    # cv = SlidingWindowSplitter(initial_window=int(len(y_train) * 0.5))
    # gscv = ForecastingGridSearchCV(forecaster, cv=cv, param_grid=param_grid)
    # gscv.fit(y_train)
    # y_pred = gscv.predict(fh)

    # plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
    # smape_loss(y_test, y_pred)

    # gscv.best_params_
# %%
############################## Solver ##############################
###################### Type series: Upward brownian, missing values sol: no missing values comparison ############################
df = pd.read_csv("./synthetic_data/brownian_scenarios/upward_mu_021_sig_065.csv")
df = df['stockprice']
y = df
# fig, ax = plot_series(y)
# ax.set(xlabel="days", ylabel="synthethic stock")

# Split the airline data into train and test split.
y_train, y_test = temporal_train_test_split(y, test_size=365)
# plot_series(y_train, y_test, labels=["y_train", "y_test"])
print("the shape of the training and test is: ", y_train.shape[0], y_test.shape[0])

# Define the x values for the prediction. 
fh = np.arange(len(y_test)) + 1

# using the naive forecaster
forecaster = NaiveForecaster(strategy="last")
forecaster.fit(y_train)
y_pred = forecaster.predict(fh)
plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
smape_naive = smape_loss(y_pred, y_test)

# using the linear forecaster
forecaster = PolynomialTrendForecaster(degree=1)
forecaster.fit(y_train)
y_pred = forecaster.predict(fh)
plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
smape_linear = smape_loss(y_pred, y_test)

# plot the detrender liner detrending
forecaster = PolynomialTrendForecaster(degree=1)
transformer = Detrender(forecaster=forecaster)
yt = transformer.fit_transform(y_train)
# internally, the Detrender uses the in-sample predictions of the PolynomialTrendForecaster
forecaster = PolynomialTrendForecaster(degree=1)
fh_ins = -np.arange(len(y_train))  # in-sample forecasting horizon
fh_ins
y_pred = forecaster.fit(y_train).predict(fh=fh_ins)
plot_series(y_train, y_pred, yt, labels=["y_train", "fitted linear trend", "residuals"])

# plot the detrender liner detrending
forecaster = AutoARIMA(sp=12, suppress_warnings=True)
forecaster.fit(y_train)
y_pred = forecaster.predict(fh)
plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
smape_ARIMA = smape_loss(y_test, y_pred)
# %%
print("For the upward stock no missing values,\n the SMAPE of the Naive forecaster is: ", "{:.3f}".format(smape_naive), "\n the linear forecaster is: ", "{:.3f}".format(smape_linear), "\n AutoArima forecaster is: ", "{:.3f}".format(smape_ARIMA))

###################### Type series: Upward brownian, missing values sol: fill with interpolation  ############################
###################### Type series: Upward brownian, missing values sol: fill with previous values ############################
# df = pd.read_csv("./synthetic_data/brownian_scenarios/upward_mu_021_sig_065.csv")
# data = df['stockprice_missing_closed_days']
# y = data
# fig, ax = plot_series(y)
# ax.set(xlabel="days", ylabel="synthethic data")


# %%
