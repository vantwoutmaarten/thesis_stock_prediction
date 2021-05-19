import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import copy


#copies the stock price, makes 1 new column 30 values ahead, makes 1 new column with 30% missing and one with 90% missing.
def create_2D_shifted_stockprice(scenario_name):

    df = web.DataReader('AAPL', data_source='yahoo', start='2016-12-31',  end='2020-12-31')
    total_days = df['Close'].count()
    data = df.filter(['Close'])

    df = data
    # Create the new column that is shifted by 30 values (postive shift means the new column lags behind. Negative shift means the new column is ahead thus knows the future of the original). 
    df_lagged = df.copy()

    shifted = df.shift(-30)
    shifted.columns = [x + "_ahead" + str(30) for x in df.columns]

    df_lagged = pd.concat((df_lagged, shifted), axis=1)

    df_lagged = df_lagged.dropna()
    # with this the index can be dropped, but then that should be done after later concatenations aswell, 
    # This should not be used when matching price columns with quarterly figure columns, since these should be concatinated by date.
    # Now this is done with numerical index, because it is easier to modify the properties such as sparseness, such as using #days as index.
    # df_lagged = df_lagged.reset_index(drop=True)
    # shifted = shifted.reset_index(drop=True)

    # This creates a new column with 30 sparseness
    close_ahead30_missing30 = df_lagged.filter(['Close_ahead30']).copy()
    close_ahead30_missing30.columns = [x + "_missing" + str(30) for x in df_lagged.filter(['Close_ahead30']).columns]

    day = 0
    for key, value in close_ahead30_missing30.iterrows():
        print(key, value)
        if(day%3 == 2):
            print("MAKE NP NAN VALUES")
            close_ahead30_missing30[key] = np.nan
        print(key, value)
        day = day + 1
        print(day)

    # while(day < total_days):
    #     if(day%3 == 2):
    #         print("this column is changed.")
    #         print(str(close_ahead30_missing30[day]))
    #         close_ahead30_missing30[day] = np.nan
    #     day = day + 1

    # close_ahead30_missing90 = shifted.copy()
    # close_ahead30_missing90.columns = [x + "_missing" + str(90) for x in shifted.columns]

    # day = 0
    # while(day < total_days):
    #     if(day%10 == 9):
    #         close_ahead30_missing90[day] = np.nan
    #     day = day + 1

    df_lagged = pd.concat((df_lagged, close_ahead30_missing30), axis=1)
    # df_lagged = df_lagged.reset_index(drop=True)

    output_loc = 'data_price/' + scenario_name

    df_lagged.to_csv(output_loc)

create_2D_shifted_stockprice(scenario_name = 'AAPL_Shifted_30ahead.csv')
