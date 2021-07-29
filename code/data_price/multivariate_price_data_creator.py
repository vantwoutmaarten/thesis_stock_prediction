import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import copy

# Save stock price data from yahoo
# when missing = True,  it creates artificial missingness that replicates the amount of missingness in quarterly figures compared to the daily frame.
def create_1D_stockprice(scenario_name, missing=False):
    df = web.DataReader('AAPL', data_source='yahoo', start='2017-03-31',  end='2021-06-01')
    total_days = df['Close'].count()
    data = df.filter(['Close'])
    
    if(missing == False):
        output_loc = 'thesis_datasets/Dataset1/' + scenario_name
        data.to_csv(output_loc)
        return

    # This creates a new column with 98.5% missingness
    close_missing985 = df.filter(['Close']).copy()
    close_missing985.columns = [x + "_missing" + str(98.5) for x in df.filter(['Close']).columns]

    day = 0
    #(Every 1st of 65 values is present, then 64/65 missing) Thus, 98.5% missingness
    for key, value in close_missing985.iterrows():
        if(day%65 != 0):
            close_missing985.loc[key,'Close_missing98.5'] = np.nan
        day = day + 1
    
    df_lagged = pd.concat((data, close_missing985), axis=1)
    output_loc = 'thesis_datasets/Dataset2/' + scenario_name

    df_lagged.to_csv(output_loc)
    return
        

    

#copies the stock price, makes 1 new column 30 values ahead, then this column is copied and made sparse in two ways.
# one way: new copied column with 30% missing. Second way:new copied column with 90% missing.
def create_2D_shifted_stockprice(scenario_name, missing_percentage = 0):

    df = web.DataReader('MSFT', data_source='yahoo', start='2017-03-31',  end='2021-07-14')
    total_days = df['Close'].count()
    data = df.filter(['Close'])

    df = data
    # Create the new column that is shifted by 30 values (postive shift means the new column lags behind. Negative shift means the new column is ahead thus knows the future of the original). 
    df_lagged = df.copy()

    shifted = df.shift(-30)
    shifted.columns = [x + "_ahead" + str(30) for x in df.columns]

    df_lagged = pd.concat((df_lagged, shifted), axis=1)

    df_lagged = df_lagged.dropna()
    print("size before missing")
    print(df_lagged.size)
    # with this the index can be dropped, but then that should be done after later concatenations aswell, 
    # This should not be used when matching price columns with quarterly figure columns, since these should be concatinated by date.
    # Now this is not done to have similarity between the test data and eventual quarterly figure data. (got the missingness working with dates soNice)
    # df_lagged = df_lagged.reset_index(drop=True)
    # shifted = shifted.reset_index(drop=True)

    if(missing_percentage == 33):
        # This creates a new column with 33% missingness
        close_ahead30_missing33 = df_lagged.filter(['Close_ahead30']).copy()
        close_ahead30_missing33.columns = [x + "_missing" + str(33) for x in df_lagged.filter(['Close_ahead30']).columns]

        day = 0
        #Every third value is missing = 33% missingness
        for key, value in close_ahead30_missing33.iterrows():
            if(day%3 == 2):
                close_ahead30_missing33.loc[key,'Close_ahead30_missing33'] = np.nan
            day = day + 1

        df_lagged = pd.concat((df_lagged, close_ahead30_missing33), axis=1)
        
        output_loc = 'thesis_datasets/Dataset3/Missing33/' + scenario_name

    if(missing_percentage == 90):
        # This creates a new column with 90 missingness
        close_ahead30_missing90 = df_lagged.filter(['Close_ahead30']).copy()
        close_ahead30_missing90.columns = [x + "_missing" + str(90) for x in df_lagged.filter(['Close_ahead30']).columns]

        day = 0
        #(Every 1st of 10 values is present, then 9/10 missing) Thus, 90% missingness
        for key, value in close_ahead30_missing90.iterrows():
            if(day%10 != 0):
                close_ahead30_missing90.loc[key,'Close_ahead30_missing90'] = np.nan
            day = day + 1
        
        df_lagged = pd.concat((df_lagged, close_ahead30_missing90), axis=1)

        output_loc = 'thesis_datasets/Dataset3/Missing90/' + scenario_name
    
    if(missing_percentage == 98.5):
        # This creates a new column with 98.5 missingness
        close_ahead30_missing985 = df_lagged.filter(['Close_ahead30']).copy()
        close_ahead30_missing985.columns = [x + "_missing" + str(98.5) for x in df_lagged.filter(['Close_ahead30']).columns]

        day = 0
        #(Every 1st of 10 values is present, then 9/10 missing) Thus, 90% missingness
        for key, value in close_ahead30_missing985.iterrows():
            if(day%65 != 0):
                close_ahead30_missing985.loc[key,'Close_ahead30_missing98.5'] = np.nan
            day = day + 1
        
        df_lagged = pd.concat((df_lagged, close_ahead30_missing985), axis=1)

        output_loc = 'thesis_datasets/Dataset3/Missing985/' + scenario_name

    df_lagged.to_csv(output_loc)
    return

# create_2D_shifted_stockprice(scenario_name = 'MSFT_Shifted_30ahead_m985.csv', missing_percentage = 98.5)

# create_1D_stockprice(scenario_name = 'AAPL.csv', missing=FALSE)