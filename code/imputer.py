import numpy as np
import pandas as pd

def createImputedColumns(df, col_name, forwardfill = False, globalmean = False):
    if(forwardfill):
        createForwardFilledColumn(df, col_name)
    
    if(globalmean):
        createGlobalMeanFilledColumn(df, col_name)
    
    df.to_csv(FILEPATH)

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

createImputedColumns(df, col_name, globalmean=True)
