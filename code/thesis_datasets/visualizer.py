import pandas as pd
import matplotlib.pyplot as plt
from sktime.utils.plotting import plot_series
import seaborn as sns
# Dataset 1
FILEPATH_AAPL = "./thesis_datasets/Dataset1/AAPL.csv"
FILEPATH_MSFT = "./thesis_datasets/Dataset1/MSFT.csv"
FILEPATH_KO = "./thesis_datasets/Dataset1/KO.csv"

df_AAPL = pd.read_csv(FILEPATH_AAPL)
df_MSFT = pd.read_csv(FILEPATH_MSFT)
df_KO = pd.read_csv(FILEPATH_KO)

df_AAPL.columns = ['Date', 'Closing Price AAPL']
df_MSFT.columns = ['Date','Closing Price MSFT']
df_KO.columns = ['Date', 'Closing Price KO']

df = pd.merge(df_AAPL, df_MSFT, how='inner')
df = pd.merge(df, df_KO, how='inner')
# # gca stands for 'get current axis'
ax = plt.gca()

df.plot(kind='line',x='Date',y='Closing Price AAPL',ax=ax)
df.plot(kind='line',x='Date',y='Closing Price MSFT',ax=ax)
df.plot(kind='line',x='Date',y='Closing Price KO',ax=ax)
plt.legend(loc='best')
plt.show()
