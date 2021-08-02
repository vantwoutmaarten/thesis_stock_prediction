import pandas as pd
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
from sklearn.preprocessing import MinMaxScaler
from torch.random import seed

FILEPATH = './thesis_datasets/Dataset4/AAPL_Price_and_Quarterly.csv'

df = pd.read_csv(FILEPATH)

data_name = 'Close'

data = df
test_size = 20
# The test size here is 20, this creates the split between what data is known and not known, like training and test.
# feature 1
data_name = 'Close'
# feature 2, 3
EnterpriseValue_meanlast260 ='EnterpriseValue_meanlast260'
EnterpriseValue_linearfit260 = 'EnterpriseValue_linearfit260'
# feature 4, 5
PeRatio_meanlast260 = 'PeRatio_meanlast260'
PeRatio_linearfit260 = 'PeRatio_linearfit260'
# feature 6, 7
ForwardPeRatio_meanlast260 = 'ForwardPeRatio_meanlast260'
ForwardPeRatio_linearfit260 = 'ForwardPeRatio_linearfit260'
# feature 8, 9
PegRatio_meanlast260 = 'PegRatio_meanlast260'
PegRatio_linearfit260 = 'PegRatio_linearfit260'
# feature 10, 11
EnterprisesValueEBITDARatio_meanlast260 = 'EnterprisesValueEBITDARatio_meanlast260'
EnterprisesValueEBITDARatio_linearfit260 = 'EnterprisesValueEBITDARatio_linearfit260'
# feature 12
time_lag = 'time_lag'

data = df.filter(items=[data_name, EnterpriseValue_meanlast260, EnterpriseValue_linearfit260, PeRatio_meanlast260,
    PeRatio_linearfit260, ForwardPeRatio_meanlast260, ForwardPeRatio_linearfit260, PegRatio_meanlast260, PegRatio_linearfit260, EnterprisesValueEBITDARatio_meanlast260, EnterprisesValueEBITDARatio_linearfit260, time_lag])

scaler = MinMaxScaler(feature_range=(-1, 1))

data_scaled = scaler.fit_transform(data)
data = pd.DataFrame(data_scaled, columns=data.columns)

y_train, y_test = temporal_train_test_split(data[data_name], test_size=test_size)

scaler = MinMaxScaler(feature_range=(-1, 1))
y_train = scaler.fit(data[data_name].values.reshape(-1,1)).transform(y_train.values.reshape(-1,1))
y_train = pd.Series(y_train.reshape(-1))
y_train.index = list(y_train.index)	

fig2, ax = plot_series(
    y_train,
    y_test,
    labels=["y_train","y_test"]
    )
