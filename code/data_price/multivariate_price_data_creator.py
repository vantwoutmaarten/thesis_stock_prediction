import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web

def create_2D_shifted_stockprice(scenario_name):

    df = web.DataReader('AAPL', data_source='yahoo', start='2016-01-01',  end='2020-11-09')
    data = df.filter(['Close'])
    print("hello")
    print(data)
    # closePrice = pd.Series(data)

    # frame = {'closePrice' : closePrice}

    df = data
    # Create the new column that is shifted by 30 values (postive shift means the new column lags behind. Negative shift means the new column is ahead thus knows the future of the original). 
    df_lagged = df.copy()

    shifted = df.shift(-30)
    shifted.columns = [x + "_ahead" + str(30) for x in df.columns]

    df_lagged = pd.concat((df_lagged, shifted), axis=1)

    df_lagged = df_lagged.dropna()

    df_lagged = df_lagged.reset_index(drop=True)
    # This creates a new column with 30 sparseness
    sin_regular_missing = df.copy(df_lagged)
    day = 0
    while(day < total_days):
        if(day%3 == 2):
            sin_regular_missing[day] = np.nan
        day = day + 1




    output_loc = 'data_price/' + scenario_name

    df_lagged.to_csv(output_loc)

create_2D_shifted_stockprice(scenario_name = 'AAPL_Shifted_30ahead.csv')
