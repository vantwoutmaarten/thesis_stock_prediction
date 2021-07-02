import numpy as np
import pandas as pd

from sktime.forecasting.trend import PolynomialTrendForecaster

def createImputedColumns(df, col_name, 
                        forwardfill = False, globalmean = False, windowof30mean = False, linear30fit = False, cubic30fit = False,
                        globalmean_incl_imputation=True, windowmean_incl_imputation=True, linear30fit_incl_imputatation=True, cubic30fit_incl_imputation=True):
    if(forwardfill):
        createForwardFilledColumn(df, col_name)
    
    if(globalmean):
        createGlobalMeanFilledColumn(df, col_name, globalmean_incl_imputation)
    
    if(windowof30mean):
        createMeanOfLast30Column(df, col_name, windowmean_incl_imputation)
    
    if(linear30fit):
        createLinear30FitColumn(df, col_name, linear30fit_incl_imputatation)
    
    if(cubic30fit):
        createCubic30FitColumn(df, col_name, cubic30fit_incl_imputation)

def createLinear30FitColumn(df, col_name, linear30fit_incl_imputatation):
    new_col_name = col_name + '_linearfit30'
    # new_col_name = col_name + '_imputed'
    
    linearfit_missing = df[col_name].copy()

    forecast = linearfit_missing[0]

    for day in range(df['Close_ahead30'].count()):
        if(np.isnan(linearfit_missing[day])):
            if(day != 0):
                if(day < 30):
                    fh = np.arange(1)+1
                    forecaster = PolynomialTrendForecaster(degree=1)
                    y_train = linearfit_missing[0:day]
                    forecaster.fit(y_train)
                    y_pred = forecaster.predict(fh)
                    linearfit_missing[day] = y_pred
                else:
                    fh = np.arange(1)
                    forecaster = PolynomialTrendForecaster(degree=1)
                    y_train = linearfit_missing[day-30:day]
                    forecaster.fit(y_train)
                    y_pred = forecaster.predict(fh)
                    linearfit_missing[day] = y_pred
        
    df[new_col_name] = linearfit_missing
    df.to_csv(FILEPATH, index=False)

def createCubic30FitColumn(df, col_name, cubic30fit_incl_imputation):
    new_col_name = col_name + '_cubicfit30'
    # new_col_name = col_name + '_imputed'

    cubicfit_missing = df[col_name].copy()

    forecast = cubicfit_missing[0]

    for day in range(df['Close_ahead30'].count()):
        if(np.isnan(cubicfit_missing[day])):
            if(day != 0):
                if(day < 30):
                    fh = np.arange(1)+1
                    forecaster = PolynomialTrendForecaster(degree=3)
                    y_train = cubicfit_missing[0:day]
                    forecaster.fit(y_train)
                    y_pred = forecaster.predict(fh)
                    cubicfit_missing[day] = y_pred
                else:
                    fh = np.arange(1)
                    forecaster = PolynomialTrendForecaster(degree=3)
                    y_train = cubicfit_missing[day-30:day]
                    forecaster.fit(y_train)
                    y_pred = forecaster.predict(fh)
                    cubicfit_missing[day] = y_pred
        
    df[new_col_name] = cubicfit_missing
    df.to_csv(FILEPATH, index=False)

def createMeanOfLast30Column(df, col_name, windowmean_incl_imputation):
    new_col_name = col_name + '_meanlast30'
    # new_col_name = col_name + '_imputed'

    windowmean_imputed = df[col_name].copy()
    windowmean_missing = df[col_name].copy()

    windowMean = windowmean_imputed[0]
    
    # Calculate the window average including imputed values
    if(windowmean_incl_imputation):
        for day in range(df['Close_ahead30'].count()):
            if(np.isnan(windowmean_imputed[day])):
                windowmean_imputed[day] = windowMean
            # Calculate the average of the first window
            # After the window size of 30, the value of 30 steps ago will be deleted 
            # and the new daily value will be added to the 30 day window mean.
            if(day != 0):
                valueOfDay = windowmean_imputed[day]
                if(day < 30):
                    windowMean = ((windowMean*(day))+valueOfDay)/(day+1)
                else: 
                    value30ago = windowmean_imputed[day-30]
                    windowMean = ((windowMean*(30))+valueOfDay-value30ago)/30
    else:
    # Calculate the window average of only real values.
        for day in range(df['Close_ahead30'].count()):
            if(np.isnan(windowmean_imputed[day])):
                windowmean_imputed[day] = windowMean
            if(day < 30):
                windowMean = windowmean_missing.loc[0:day].mean()
            else:
                windowMean = windowmean_missing.loc[day-29:day].mean()

    df[new_col_name] = windowmean_imputed
    df.to_csv(FILEPATH, index=False)

def createGlobalMeanFilledColumn(df, col_name, globalmean_incl_imputation):
    new_col_name = col_name + '_globalmean'
    # new_col_name = col_name + '_imputed'

    globalmean_missing = df[col_name].copy()

    globalMean = globalmean_missing[0]
    totalRealValues = 1
    
    # Calculate the running average including imputed values
    if(globalmean_incl_imputation):
        for day in range(df['Close_ahead30'].count()):
                if(np.isnan(globalmean_missing[day])):
                    globalmean_missing[day] = globalMean
                if(day != 0):
                    valueOfDay = globalmean_missing[day]
                    globalMean = ((globalMean*(day))+valueOfDay)/(day+1)
    else:
    # Calculate the running average of only actual values.
        for day in range(df['Close_ahead30'].count()):
            if(np.isnan(globalmean_missing[day])):
                globalmean_missing[day] = globalMean
            elif(day != 0):
                valueOfDay = globalmean_missing[day]
                globalMean = ((globalMean*(totalRealValues))+valueOfDay)/(totalRealValues+1)
                totalRealValues += 1

    df[new_col_name] = globalmean_missing
    df.to_csv(FILEPATH, index=False)

def createForwardFilledColumn(df, col_name):
    new_col_name = col_name + '_forwardfill'
    # new_col_name = col_name + '_imputed'

    forwardfill_missing = df[col_name].copy()

    # This method does forward filling, at the moment it is only compatible with series with a value at index 0
    print(df['Close_ahead30'].count())
    for day in range(df['Close_ahead30'].count()):
        if(np.isnan(forwardfill_missing[day])):
            forwardfill_missing[day] = forwardfill_missing[day-1]
            print(forwardfill_missing[day])

    df[new_col_name] = forwardfill_missing
    df.to_csv(FILEPATH, index=False) 

# FILEPATH = "./synthetic_data/univariate_missingness/noisy_sin_period126_seasonalperiod628_year7_missing33_seed2.csv"
# FILEPATH = "./data_price/data/Apple/missing33/AAPL_Shifted_30ahead.csv"
FILEPATH = "imputatationtestfile.csv"
df = pd.read_csv(FILEPATH)
# df = pd.read_csv(FILEPATH, index_col=0)
# col_name = 'noisy_sin_random_missing'
col_name = 'Close_ahead30_missing30'

createImputedColumns(df, col_name, 
                    forwardfill = False, globalmean = False, windowof30mean = True, linear30fit = False, cubic30fit = False,
                    globalmean_incl_imputation=False, windowmean_incl_imputation=False, linear30fit_incl_imputatation=False, cubic30fit_incl_imputation=False)