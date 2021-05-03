import numpy as np
import pandas as pd

def createImputedColumns(df, col_name, forwardfill = False, globalmean = False, windowof30mean= False):
    if(forwardfill):
        createForwardFilledColumn(df, col_name)
    
    if(globalmean):
        createGlobalMeanFilledColumn(df, col_name)
    
    if(windowof30mean):
        createMeanOfLast30Column(df, col_name)
    
    df.to_csv(FILEPATH)

def createMeanOfLast30Column(df, col_name):
    new_col_name = col_name + '_meanlast30'
    windowmean_missing = df[col_name].copy()

    windowMean = windowmean_missing[0]

    for day in range(df['noisy_sin'].count()):

        if(np.isnan(windowmean_missing[day])):
            windowmean_missing[day] = windowMean

        # Calculate the average of the first window
        # After the window size of 30, the value of 30 steps ago will be deleted 
        # and the new daily value will be added to the 30 day window mean.

        if(day != 0):
            valueOfDay = windowmean_missing[day]
            if(day < 30):
                windowMean = ((windowMean*(day))+valueOfDay)/(day+1)
            else: 
                value30ago = windowmean_missing[day-30]
                windowMean = ((windowMean*(30))+valueOfDay-value30ago)/30
    
    df[new_col_name] = windowmean_missing
    df.to_csv(FILEPATH, index=False)

def createGlobalMeanFilledColumn(df, col_name):
    new_col_name = col_name + '_globalmean'
    globalmean_missing = df[col_name].copy()

    globalMean = globalmean_missing[0]
    
    for day in range(df['noisy_sin'].count()):

        if(np.isnan(globalmean_missing[day])):
            globalmean_missing[day] = globalMean

        # Calculate the running average
        if(day != 0):
            valueOfDay = globalmean_missing[day]
            # when starting from index 1, ((globalMean*(day-1))+valueOfDay)/day
            globalMean = ((globalMean*(day))+valueOfDay)/(day+1)

    df[new_col_name] = globalmean_missing
    df.to_csv(FILEPATH, index=False)


def createForwardFilledColumn(df, col_name):
    new_col_name = col_name + '_forwardfill'
    forwardfill_missing = df[col_name].copy()

    # This method does forward filling, at the moment it is only compatible with series with a value at index 0
    print(df['noisy_sin'].count())
    for day in range(df['noisy_sin'].count()):
        if(np.isnan(forwardfill_missing[day])):
            forwardfill_missing[day] = forwardfill_missing[day-1]
            print(forwardfill_missing[day])

    df[new_col_name] = forwardfill_missing
    df.to_csv(FILEPATH, index=False)

FILEPATH = "./synthetic_data/univariate_missingness/test.csv"
df = pd.read_csv(FILEPATH, index_col=0)
# col_name = 'noisy_sin_random_missing'
col_name = 'noisy_sin_regular_missing'

createImputedColumns(df, col_name, globalmean=False, forwardfill=False, windowof30mean= True)
