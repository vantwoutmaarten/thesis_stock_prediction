import pandas as pd

FILEPATH_QUARTERLY= "./thesis_datasets/Dataset5/MSFT_quarterly_valuation_measures.csv"
FILEPATH_PRICE = "./thesis_datasets/Dataset5/MSFT_price.csv"

dfQ = pd.read_csv(FILEPATH_QUARTERLY)
dfP = pd.read_csv(FILEPATH_PRICE)


dfQ.set_index('name',inplace=True)
dfQ = dfQ.transpose()

dfQ = dfQ.iloc[1:19]

dfQ.index.name = 'Date'
dfP.set_index('Date',inplace=True)
dfP.index.name = 'Date'

dfP.index = pd.to_datetime(dfP.index).date
dfQ.index = pd.to_datetime(dfQ.index).date

dfQ = dfQ.filter(items = ['EnterpriseValue', 'PeRatio', 'ForwardPeRatio', 'PegRatio','EnterprisesValueEBITDARatio'])

# df_combined = pd.concat([dfP, dfQ], join='left', axis=1)

df_combined = dfP.join(dfQ, how='outer')
df_combined.index.name = 'Date'

FILEPATH_OUTPUT = "./thesis_datasets/Dataset5/MSFT_Price_and_Quarterly.csv"

df_combined.to_csv(FILEPATH_OUTPUT) 
