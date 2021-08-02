from os import truncate
import numpy as np
import pandas as pd

from sktime.forecasting.trend import PolynomialTrendForecaster
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import interp1d

def createImputedColumns(df, col_name, 
                        forwardfill = False, globalmean = False, windowof260mean = False, linear260fit = False, cubic260fit = False,
                        globalmean_incl_imputation=True, windowmean_incl_imputation=True, linear260fit_incl_imputation=True, cubic260fit_incl_imputation=True):
    if(forwardfill):
        createForwardFilledColumn(df, col_name)
    
    if(globalmean):
        createGlobalMeanFilledColumn(df, col_name, globalmean_incl_imputation)
    
    if(windowof260mean):
        createMeanOfLast260Column(df, col_name, windowmean_incl_imputation)
    
    if(linear260fit):
        createLinear260FitColumn(df, col_name, linear260fit_incl_imputation)
    
    if(cubic260fit):
        createCubic260FitColumn(df, col_name, cubic260fit_incl_imputation)

def createLinear260FitColumn(df, col_name, linear260fit_incl_imputation):
    new_col_name = col_name + '_linearfit260'
    # new_col_name = 'Close_shifted'+  '_linearfit260'
    # new_col_name = col_name + '_imputed'
    degree = 1

    linearfit_missing = df[col_name].copy()
    linearfit_imputed = df[col_name].copy()

    forecast = linearfit_missing[0]

    for day in range(df['Date'].count()):
        if(np.isnan(linearfit_missing[day])):
            if(day != 0):
                if(day < 260):
                    if(linear260fit_incl_imputation):
                        y_train = linearfit_imputed[0:day]
                    else:
                        y_train = linearfit_missing[0:day]
                    y_train = y_train.dropna()
                    degree = 1   
                    while(len(y_train)<=degree):
                        degree = degree - 1
                    weights = np.polyfit(y_train.index.values, y_train.values, degree)
                    model = np.poly1d(weights)
                    y_pred = model(day)
                    linearfit_imputed[day] = y_pred
                else:
                    if(linear260fit_incl_imputation):
                        y_train = linearfit_imputed[day-260:day]
                    else:
                        y_train = linearfit_missing[day-260:day]
                    y_train = y_train.dropna()
                    degree = 1
                    while(len(y_train)<=degree):
                        degree = degree - 1
                    weights = np.polyfit(y_train.index.values, y_train.values, degree)
                    model = np.poly1d(weights)
                    y_pred = model(day)
                    linearfit_imputed[day] = y_pred
        
    df[new_col_name] = linearfit_imputed

    df.to_csv(FILEPATH, index=False)

def createCubic260FitColumn(df, col_name, cubic260fit_incl_imputation):
    new_col_name = col_name + '_cubicfit260'
    # new_col_name = 'Close_shifted'+  '_cubicfit260'
    # new_col_name = col_name + '_imputed'
    degree = 3

    cubicfit_missing = df[col_name].copy()
    cubicfit_imputed = df[col_name].copy()


    for day in range(df['Date'].count()):
        if(np.isnan(cubicfit_missing[day])):
            if(day != 0):
                if(day < 260):
                    if(cubic260fit_incl_imputation):
                        y_train = cubicfit_imputed[0:day]
                    else:
                        y_train = cubicfit_missing[0:day]
                    y_train = y_train.dropna()
                    degree = 3   
                    while(len(y_train)<=degree):
                        degree = degree - 1
                    weights = np.polyfit(y_train.index.values, y_train.values, degree)
                    model = np.poly1d(weights)
                    y_pred = model(day)
                    cubicfit_imputed[day] = y_pred
                else:
                    if(cubic260fit_incl_imputation):
                        y_train = cubicfit_imputed[day-260:day]
                    else:
                        y_train = cubicfit_missing[day-260:day]
                    y_train = y_train.dropna()
                    degree = 3
                    while(len(y_train)<=degree):
                        degree = degree - 1
                    weights = np.polyfit(y_train.index.values, y_train.values, degree)
                    model = np.poly1d(weights)
                    y_pred = model(day)
                    cubicfit_imputed[day] = y_pred
        
    df[new_col_name] = cubicfit_imputed

    df.to_csv(FILEPATH, index=False)

def createMeanOfLast260Column(df, col_name, windowmean_incl_imputation):
    new_col_name = col_name + '_meanlast260'
    # new_col_name = 'Close_shifted'+  '_meanlast260'
    # new_col_name = col_name + '_imputed'

    windowmean_imputed = df[col_name].copy()
    windowmean_missing = df[col_name].copy()

    windowMean = windowmean_imputed[0]
    
    # Calculate the window average including imputed values
    if(windowmean_incl_imputation):
        for day in range(df['Date'].count()):
            if(np.isnan(windowmean_imputed[day])):
                windowmean_imputed[day] = windowMean
            # Calculate the average of the first window
            # After the window size of 260, the value of 260 steps ago will be deleted 
            # and the new daily value will be added to the 260 day window mean.
            if(day != 0):
                valueOfDay = windowmean_imputed[day]
                if(day < 260):
                    windowMean = ((windowMean*(day))+valueOfDay)/(day+1)
                else: 
                    value260ago = windowmean_imputed[day-260]
                    windowMean = ((windowMean*(260))+valueOfDay-value260ago)/260
    else:
    # Calculate the window average of only real values.
        for day in range(df['Date'].count()):
            if(np.isnan(windowmean_imputed[day])):
                windowmean_imputed[day] = windowMean
            if(day < 260):
                windowMean = windowmean_missing.loc[0:day].mean()
            else:
                windowMean = windowmean_missing.loc[day-259:day].mean()

    df[new_col_name] = windowmean_imputed
    df.to_csv(FILEPATH, index=False)

def createGlobalMeanFilledColumn(df, col_name, globalmean_incl_imputation):
    new_col_name = col_name + '_globalmean'
    # new_col_name = 'Close_shifted'+ '_globalmean'
    # new_col_name = col_name + '_imputed'

    globalmean_missing = df[col_name].copy()

    globalMean = globalmean_missing[0]
    totalRealValues = 1
    
    # Calculate the running average including imputed values
    if(globalmean_incl_imputation):
        for day in range(df['Date'].count()):
                if(np.isnan(globalmean_missing[day])):
                    globalmean_missing[day] = globalMean
                if(day != 0):
                    valueOfDay = globalmean_missing[day]
                    globalMean = ((globalMean*(day))+valueOfDay)/(day+1)
    else:
    # Calculate the running average of only actual values.
        for day in range(df['Date'].count()):
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
    # new_col_name = 'Close_shifted'+ '_forwardfill'
    # new_col_name = col_name + '_imputed'

    forwardfill_missing = df[col_name].copy()

    # This method does forward filling, at the moment it is only compatible with series with a value at index 0
    for day in range(df['Date'].count()):
        if(np.isnan(forwardfill_missing[day])):
            forwardfill_missing[day] = forwardfill_missing[day-1]

    df[new_col_name] = forwardfill_missing
    df.to_csv(FILEPATH, index=False) 


def createMissingfeatures(df, col_name):

    df['missing-1or1'] = ""
    new_col_missing = df['missing-1or1']

    column_to_impute = df[col_name].copy()

    for day in range(df['Date'].count()):
            if(np.isnan(column_to_impute[day])):
                new_col_missing[day] = -1
            else:
                new_col_missing[day] = 1

    df['time_lag'] = ""
    new_col_time_lag = df['time_lag']
    column_to_impute = df[col_name].copy()
    for day in range(df['Date'].count()):
        if(np.isnan(column_to_impute[day])):
            days_since_last_observation = days_since_last_observation + 1
            new_col_time_lag[day] = days_since_last_observation
        else:
            days_since_last_observation = 0
            new_col_time_lag[day] = 0
    

    df.to_csv(FILEPATH, index=False) 



# FILEPATH = "./synthetic_data/univariate_missingness/noisy_sin_period126_seasonalperiod628_year7_missing33_seed2.csv"
# FILEPATH = "./data_price/imputed_data/Microsoft/missing90/MSFT_Shifted_30ahead.csv"
FILEPATH = "./thesis_datasets/Dataset4/KO_Price_and_Quarterly.csv"

df = pd.read_csv(FILEPATH, thousands=',')
# df = pd.read_csv(FILEPATH, index_col=0)
# col_name = 'noisy_sin_random_missing'
# col_name = 'Close_ahead30_missing90'
# col_name = 'Close_ahead30_missing98.5'
col_name1 = 'EnterpriseValue'
col_name2 = 'PeRatio'
col_name3 = 'ForwardPeRatio'
col_name4 = 'PegRatio'
col_name5 = 'EnterprisesValueEBITDARatio'


createImputedColumns(df, col_name1, 
                    forwardfill = True, globalmean = True, windowof260mean = True, linear260fit = True, cubic260fit = True,
                    globalmean_incl_imputation=False, windowmean_incl_imputation=False, linear260fit_incl_imputation=False, cubic260fit_incl_imputation=False)
createImputedColumns(df, col_name2, 
                    forwardfill = True, globalmean = True, windowof260mean = True, linear260fit = True, cubic260fit = True,
                    globalmean_incl_imputation=False, windowmean_incl_imputation=False, linear260fit_incl_imputation=False, cubic260fit_incl_imputation=False)
createImputedColumns(df, col_name3, 
                    forwardfill = True, globalmean = True, windowof260mean = True, linear260fit = True, cubic260fit = True,
                    globalmean_incl_imputation=False, windowmean_incl_imputation=False, linear260fit_incl_imputation=False, cubic260fit_incl_imputation=False)
createImputedColumns(df, col_name4, 
                    forwardfill = True, globalmean = True, windowof260mean = True, linear260fit = True, cubic260fit = True,
                    globalmean_incl_imputation=False, windowmean_incl_imputation=False, linear260fit_incl_imputation=False, cubic260fit_incl_imputation=False)
createImputedColumns(df, col_name5, 
                    forwardfill = True, globalmean = True, windowof260mean = True, linear260fit = True, cubic260fit = True,
                    globalmean_incl_imputation=False, windowmean_incl_imputation=False, linear260fit_incl_imputation=False, cubic260fit_incl_imputation=False)


createMissingfeatures(df, col_name1)