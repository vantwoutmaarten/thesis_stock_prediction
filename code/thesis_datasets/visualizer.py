from matplotlib import markers
import pandas as pd
import matplotlib.pyplot as plt
from sktime.utils.plotting import plot_series
import seaborn as sns
from pylab import rcParams
from sklearn.preprocessing import MinMaxScaler
from sktime.forecasting.model_selection import temporal_train_test_split

# Dataset 1
def plot_dataset1():
    FILEPATH_AAPL = "./thesis_datasets/Dataset1/AAPL.csv"
    FILEPATH_MSFT = "./thesis_datasets/Dataset1/MSFT.csv"
    FILEPATH_KO = "./thesis_datasets/Dataset1/KO.csv"

    rcParams['figure.figsize'] = 10, 5

    df_AAPL = pd.read_csv(FILEPATH_AAPL)
    df_MSFT = pd.read_csv(FILEPATH_MSFT)
    df_KO = pd.read_csv(FILEPATH_KO)

    df_AAPL.columns = ['Date', 'AAPL']
    df_MSFT.columns = ['Date','MSFT']
    df_KO.columns = ['Date', 'KO']

    df = pd.merge(df_AAPL, df_MSFT, how='inner')
    df = pd.merge(df, df_KO, how='inner')
    # # gca stands for 'get current axis'
    ax = plt.gca()
    plt.title("Daily closing stock prices")
    plt.ylabel("Closing price($)")
    df.plot(kind='line',x='Date',y='MSFT',ax=ax, color='#2A9D8F')
    df.plot(kind='line',x='Date',y='AAPL',ax=ax, color='#F4A261')
    df.plot(kind='line',x='Date',y='KO',ax=ax, color='#E76F51')
    plt.legend(loc='best')
    plt.savefig('./thesis_datasets/images/dataset1.png')
    plt.show()
    

def plot_dataset2():
    FILEPATH_AAPL = "./thesis_datasets/Dataset2/AAPL_missing985.csv"
    df_AAPL = pd.read_csv(FILEPATH_AAPL)
    df_AAPL.columns = ['Date','Close','missing98.5','forwardfill','globalmean','meanlast260','linearfit260','cubicfit260']

    ax = plt.gca()
    plt.title("Imputed daily AAPL stock price with 98.5% missing values")
    plt.ylabel("Price($)")
    df_AAPL.plot(kind='line',x='Date',y='forwardfill',ax=ax, color='#E76F51') 
    df_AAPL.plot(kind='line',x='Date',y='globalmean',ax=ax, color='#F4A261') 
    df_AAPL.plot(kind='line',x='Date',y='meanlast260',ax=ax, color='#E9C46A')
    df_AAPL.plot(kind='line',x='Date',y='linearfit260',ax=ax, color='#2A9D8F')
    df_AAPL.plot(kind='line',x='Date',y='cubicfit260',ax=ax, color='#264653')
    plt.legend(loc='best')
    plt.savefig('./thesis_datasets/images/dataset2_AAPL.png')
    plt.show()

    FILEPATH_MSFT = "./thesis_datasets/Dataset2/MSFT_missing985.csv"
    df_MSFT = pd.read_csv(FILEPATH_MSFT)
    df_MSFT.columns = ['Date','Close','missing98.5','forwardfill','globalmean','meanlast260','linearfit260','cubicfit260']

    ax = plt.gca()
    plt.title("Imputed daily MSFT stock price with 98.5% missing values")
    plt.ylabel("Price($)")
    df_MSFT.plot(kind='line',x='Date',y='forwardfill',ax=ax, color='#E76F51') 
    df_MSFT.plot(kind='line',x='Date',y='globalmean',ax=ax, color='#F4A261') 
    df_MSFT.plot(kind='line',x='Date',y='meanlast260',ax=ax, color='#E9C46A')
    df_MSFT.plot(kind='line',x='Date',y='linearfit260',ax=ax, color='#2A9D8F')
    df_MSFT.plot(kind='line',x='Date',y='cubicfit260',ax=ax, color='#264653')
    plt.legend(loc='best')
    plt.savefig('./thesis_datasets/images/dataset2_MSFT.png')
    plt.show()

    FILEPATH_KO = "./thesis_datasets/Dataset2/KO_missing985.csv"
    df_KO = pd.read_csv(FILEPATH_KO)
    df_KO.columns = ['Date','Close','missing98.5','forwardfill','globalmean','meanlast260','linearfit260','cubicfit260']

    ax = plt.gca()
    plt.title("Imputed daily KO stock price with 98.5% missing values")
    plt.ylabel("Price($)")
    df_KO.plot(kind='line',x='Date',y='forwardfill',ax=ax, color='#E76F51') 
    df_KO.plot(kind='line',x='Date',y='globalmean',ax=ax, color='#F4A261') 
    df_KO.plot(kind='line',x='Date',y='meanlast260',ax=ax, color='#E9C46A')
    df_KO.plot(kind='line',x='Date',y='linearfit260',ax=ax, color='#2A9D8F')
    df_KO.plot(kind='line',x='Date',y='cubicfit260',ax=ax, color='#264653')
    plt.legend(loc='best')
    plt.savefig('./thesis_datasets/images/dataset2_KO.png')

    plt.show()
        
def plot_dataset2_subplots():
    FILEPATH_AAPL = "./thesis_datasets/Dataset2/AAPL_missing985.csv"
    df_AAPL = pd.read_csv(FILEPATH_AAPL)
    df_AAPL.columns = ['Date','Close','missing98.5','forwardfill','globalmean','meanlast260','linearfit260','cubicfit260']
    rcParams['figure.figsize'] = 10, 15

    fig, axes = plt.subplots(nrows=3, ncols=1)
    fig.tight_layout()
    fig.subplots_adjust(top=0.95, bottom=0.05, left=0.06)
    plt.subplots_adjust(hspace=0.2)
    # ax = plt.gca()
    # plt.title("Imputed daily stock price with 98.5% missing values")
    fig.suptitle("Imputed daily stock price with 98.5% missing values")
    axes[0].set_title("AAPL")
    
    axes[0].set_ylabel("Price($)")
    df_AAPL.plot(kind='line',x='Date',y='forwardfill',ax=axes[0], color='#E76F51') 
    df_AAPL.plot(kind='line',x='Date',y='globalmean',ax=axes[0], color='#F4A261') 
    df_AAPL.plot(kind='line',x='Date',y='meanlast260',ax=axes[0], color='#E9C46A')
    df_AAPL.plot(kind='line',x='Date',y='linearfit260',ax=axes[0], color='#2A9D8F')
    df_AAPL.plot(kind='line',x='Date',y='cubicfit260',ax=axes[0], color='#264653')

    FILEPATH_MSFT = "./thesis_datasets/Dataset2/MSFT_missing985.csv"
    df_MSFT = pd.read_csv(FILEPATH_MSFT)
    df_MSFT.columns = ['Date','Close','missing98.5','forwardfill','globalmean','meanlast260','linearfit260','cubicfit260']

    axes[1].set_title("MSFT")
    axes[1].set_ylabel("Price($)")
    df_MSFT.plot(kind='line',x='Date',y='forwardfill',ax=axes[1], color='#E76F51') 
    df_MSFT.plot(kind='line',x='Date',y='globalmean',ax=axes[1], color='#F4A261') 
    df_MSFT.plot(kind='line',x='Date',y='meanlast260',ax=axes[1], color='#E9C46A')
    df_MSFT.plot(kind='line',x='Date',y='linearfit260',ax=axes[1], color='#2A9D8F')
    df_MSFT.plot(kind='line',x='Date',y='cubicfit260',ax=axes[1], color='#264653')

    FILEPATH_KO = "./thesis_datasets/Dataset2/KO_missing985.csv"
    df_KO = pd.read_csv(FILEPATH_KO)
    df_KO.columns = ['Date','Close','missing98.5','forwardfill','globalmean','meanlast260','linearfit260','cubicfit260']

    axes[2].set_title("KO")
    axes[2].set_ylabel("Price($)")
    df_KO.plot(kind='line',x='Date',y='forwardfill',ax=axes[2], color='#E76F51') 
    df_KO.plot(kind='line',x='Date',y='globalmean',ax=axes[2], color='#F4A261') 
    df_KO.plot(kind='line',x='Date',y='meanlast260',ax=axes[2], color='#E9C46A')
    df_KO.plot(kind='line',x='Date',y='linearfit260',ax=axes[2], color='#2A9D8F')
    df_KO.plot(kind='line',x='Date',y='cubicfit260',ax=axes[2], color='#264653')
    plt.legend(loc='upper left')

    plt.savefig('./thesis_datasets/images/dataset2_subplots.png')

    plt.show()
        

def plot_dataset3_subplots(): 
    FILEPATH_AAPL_33 = "./thesis_datasets/Dataset3/Missing33/AAPL_Shifted_30ahead_m33.csv"
    FILEPATH_AAPL_90 = "./thesis_datasets/Dataset3/Missing90/AAPL_Shifted_30ahead_m90.csv"
    FILEPATH_AAPL_985 = "./thesis_datasets/Dataset3/Missing985/AAPL_Shifted_30ahead_m985.csv"

    df_AAPL_33 = pd.read_csv(FILEPATH_AAPL_33)
    df_AAPL_90 = pd.read_csv(FILEPATH_AAPL_90)
    df_AAPL_985 = pd.read_csv(FILEPATH_AAPL_985)

    FILEPATH_MSFT_33 = "./thesis_datasets/Dataset3/Missing33/MSFT_Shifted_30ahead_m33.csv"
    FILEPATH_MSFT_90 = "./thesis_datasets/Dataset3/Missing90/MSFT_Shifted_30ahead_m90.csv"
    FILEPATH_MSFT_985 = "./thesis_datasets/Dataset3/Missing985/MSFT_Shifted_30ahead_m985.csv"

    df_MSFT_33 = pd.read_csv(FILEPATH_MSFT_33)
    df_MSFT_90 = pd.read_csv(FILEPATH_MSFT_90)
    df_MSFT_985 = pd.read_csv(FILEPATH_MSFT_985)

    FILEPATH_KO_33 = "./thesis_datasets/Dataset3/Missing33/KO_Shifted_30ahead_m33.csv"
    FILEPATH_KO_90 = "./thesis_datasets/Dataset3/Missing90/KO_Shifted_30ahead_m90.csv"
    FILEPATH_KO_985 = "./thesis_datasets/Dataset3/Missing985/KO_Shifted_30ahead_m985.csv"

    df_KO_33 = pd.read_csv(FILEPATH_KO_33)
    df_KO_90 = pd.read_csv(FILEPATH_KO_90)
    df_KO_985 = pd.read_csv(FILEPATH_KO_985)


    rcParams['figure.figsize'] = 15, 15

    fig, axes = plt.subplots(nrows=3, ncols=3)
    fig.tight_layout()
    fig.subplots_adjust(top=0.95, bottom=0.05, left=0.06)
    plt.subplots_adjust(hspace=0.2)
    # ax = plt.gca()

    fig.suptitle("Daily stock price with shifted ahead version with missingness")
    
    axes[0,0].set_title("APPL,Shifted price 33% missingness")
    df_AAPL_33.reset_index().plot(kind='line',x='index',y='Close',ax=axes[0,0], color='#E76F51') 
    df_AAPL_33.reset_index().plot(kind='scatter',x='index',y='Close_ahead30_missing33',ax=axes[0,0], color='#2A9D8F', s=2.0) 
    axes[0,0].plot([], [], 'o', color='#2A9D8F', label = 'Close_ahead30_missing33')
    axes[0,0].legend()
    axes[0,0].set_ylabel("Price($)")

    axes[1,0].set_title("APPL,Shifted price 90% missingness")
    df_AAPL_90.reset_index().plot(kind='line',x='index',y='Close',ax=axes[1,0], color='#E76F51') 
    df_AAPL_90.reset_index().plot(kind='scatter',x='index',y='Close_ahead30_missing90',ax=axes[1,0], color='#2A9D8F', s=3.0) 
    axes[1,0].plot([], [], 'o', color='#2A9D8F', label = 'Close_ahead30_missing90')
    axes[1,0].legend()
    axes[1,0].set_ylabel("Price($)")

    axes[2,0].set_title("APPL,Shifted price 98.5% missingness")
    df_AAPL_985.reset_index().plot(kind='line',x='index',y='Close',ax=axes[2,0], color='#E76F51') 
    df_AAPL_985.reset_index().plot(kind='scatter',x='index',y='Close_ahead30_missing98.5',ax=axes[2,0], color='#2A9D8F', s=6.0) 
    axes[2,0].plot([], [], 'o', color='#2A9D8F', label = 'Close_ahead30_missing98.5')
    axes[2,0].legend()
    axes[2,0].set_ylabel("Price($)")

    axes[0,1].set_title("MSFT,Shifted price 33% missingness")
    df_MSFT_33.reset_index().plot(kind='line',x='index',y='Close',ax=axes[0,1], color='#E76F51') 
    df_MSFT_33.reset_index().plot(kind='scatter',x='index',y='Close_ahead30_missing33',ax=axes[0,1], color='#2A9D8F', s=2.0) 
    axes[0,1].plot([], [], 'o', color='#2A9D8F', label = 'Close_ahead30_missing33')
    axes[0,1].legend()
    axes[0,1].set_ylabel("Price($)")

    axes[1,1].set_title("MSFT,Shifted price 90% missingness")
    df_MSFT_90.reset_index().plot(kind='line',x='index',y='Close',ax=axes[1,1], color='#E76F51') 
    df_MSFT_90.reset_index().plot(kind='scatter',x='index',y='Close_ahead30_missing90',ax=axes[1,1], color='#2A9D8F', s=3.0) 
    axes[1,1].plot([], [], 'o', color='#2A9D8F', label = 'Close_ahead30_missing90')
    axes[1,1].legend()
    axes[1,1].set_ylabel("Price($)")

    axes[2,1].set_title("MSFT,Shifted price 98.5% missingness")
    df_MSFT_985.reset_index().plot(kind='line',x='index',y='Close',ax=axes[2,1], color='#E76F51') 
    df_MSFT_985.reset_index().plot(kind='scatter',x='index',y='Close_ahead30_missing98.5',ax=axes[2,1], color='#2A9D8F', s=6.0) 
    axes[2,1].plot([], [], 'o', color='#2A9D8F', label = 'Close_ahead30_missing98.5')
    axes[2,1].legend()
    axes[2,1].set_ylabel("Price($)")
    
    axes[0,2].set_title("KO,Shifted price 33% missingness")
    df_KO_33.reset_index().plot(kind='line',x='index',y='Close',ax=axes[0,2], color='#E76F51') 
    df_KO_33.reset_index().plot(kind='scatter',x='index',y='Close_ahead30_missing33',ax=axes[0,2], color='#2A9D8F', s=2.0) 
    axes[0,2].plot([], [], 'o', color='#2A9D8F', label = 'Close_ahead30_missing33')
    axes[0,2].legend()
    axes[0,2].set_ylabel("Price($)")

    axes[1,2].set_title("KO,Shifted price 90% missingness")
    df_KO_90.reset_index().plot(kind='line',x='index',y='Close',ax=axes[1,2], color='#E76F51') 
    df_KO_90.reset_index().plot(kind='scatter',x='index',y='Close_ahead30_missing90',ax=axes[1,2], color='#2A9D8F', s=6.0) 
    axes[1,2].plot([], [], 'o', color='#2A9D8F', label = 'Close_ahead30_missing90')
    axes[1,2].legend()
    axes[1,2].set_ylabel("Price($)")

    axes[2,2].set_title("KO,Shifted price 98.5% missingness")
    df_KO_985.reset_index().plot(kind='scatter',x='index',y='Close_ahead30_missing98.5',ax=axes[2,2], color='#2A9D8F', s=6.0)
    df_KO_985.reset_index().plot(kind='line',x='index',y='Close',ax=axes[2,2], color='#E76F51') 
    axes[2,2].plot([], [], 'o', color='#2A9D8F', label = 'Close_ahead30_missing98.5')
    axes[2,2].legend()
    axes[2,2].set_ylabel("Price($)")

    plt.savefig('./thesis_datasets/images/dataset3_subs_missingness.png')

    plt.legend(loc='best')
    plt.show()
    
def plot_dataset4_quarterly():
    FILEPATH_AAPL = "./thesis_datasets/Dataset4/AAPL_Price_and_Quarterly.csv"
    df_AAPL = pd.read_csv(FILEPATH_AAPL)
    FILEPATH_MSFT = "./thesis_datasets/Dataset4/MSFT_Price_and_Quarterly.csv"
    df_MSFT = pd.read_csv(FILEPATH_MSFT)
    FILEPATH_KO = "./thesis_datasets/Dataset4/KO_Price_and_Quarterly.csv"
    df_KO = pd.read_csv(FILEPATH_KO)

    rcParams['figure.figsize'] = 10, 15

    fig, axes = plt.subplots(nrows=5, ncols=1)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.subplots_adjust(top=0.95, bottom=0.05, left=0.06)
    plt.subplots_adjust(hspace=0.2)
    # ax = plt.gca()

    fig.suptitle("Quarterly valuation numbers of three stocks")

    axes[0].set_title("Enterprise Value")
    df_AAPL.reset_index().plot(kind='scatter',x='index',y='EnterpriseValue',ax=axes[0], color='#2A9D8F') 
    df_MSFT.reset_index().plot(kind='scatter',x='index',y='EnterpriseValue',ax=axes[0], color='#E9C46A') 
    df_KO.reset_index().plot(kind='scatter',x='index',y='EnterpriseValue',ax=axes[0], color='#E76F51')
    axes[0].plot([], [], 'o', color='#2A9D8F', label = 'AAPL')
    axes[0].plot([], [], 'o', color='#E9C46A', label = 'MSFT')
    axes[0].plot([], [], 'o', color='#E76F51', label = 'KO')
    axes[0].set_ylabel("EnterpriseValue($)")
    axes[0].legend()

    axes[1].set_title("Trailing P/E-ratio")
    df_AAPL.reset_index().plot(kind='scatter',x='index',y='PeRatio',ax=axes[1], color='#2A9D8F') 
    df_MSFT.reset_index().plot(kind='scatter',x='index',y='PeRatio',ax=axes[1], color='#E9C46A') 
    df_KO.reset_index().plot(kind='scatter',x='index',y='PeRatio',ax=axes[1], color='#E76F51')
    axes[1].plot([], [], 'o', color='#2A9D8F', label = 'AAPL')
    axes[1].plot([], [], 'o', color='#E9C46A', label = 'MSFT')
    axes[1].plot([], [], 'o', color='#E76F51', label = 'KO')
    axes[1].set_ylabel("Trailing P/E(-)")


    axes[2].set_title("Forward P/E-ratio")
    df_AAPL.reset_index().plot(kind='scatter',x='index',y='ForwardPeRatio',ax=axes[2], color='#2A9D8F') 
    df_MSFT.reset_index().plot(kind='scatter',x='index',y='ForwardPeRatio',ax=axes[2], color='#E9C46A') 
    df_KO.reset_index().plot(kind='scatter',x='index',y='ForwardPeRatio',ax=axes[2], color='#E76F51')
    axes[2].plot([], [], 'o', color='#2A9D8F', label = 'AAPL')
    axes[2].plot([], [], 'o', color='#E9C46A', label = 'MSFT')
    axes[2].plot([], [], 'o', color='#E76F51', label = 'KO')
    axes[2].set_ylabel("Forward P/E(-)")


    axes[3].set_title("PEG-ratio (5 yr expected)")
    df_AAPL.reset_index().plot(kind='scatter',x='index',y='PegRatio',ax=axes[3], color='#2A9D8F') 
    df_MSFT.reset_index().plot(kind='scatter',x='index',y='PegRatio',ax=axes[3], color='#E9C46A') 
    df_KO.reset_index().plot(kind='scatter',x='index',y='PegRatio',ax=axes[3], color='#E76F51')
    axes[3].plot([], [], 'o', color='#2A9D8F', label = 'AAPL')
    axes[3].plot([], [], 'o', color='#E9C46A', label = 'MSFT')
    axes[3].plot([], [], 'o', color='#E76F51', label = 'KO')
    axes[3].set_ylabel("PEG(-)")


    axes[4].set_title("Enterprise Value/EBITDA-ratio")

    df_AAPL["empty"] = ""
    # df_AAPL.plot(kind='line',x='Date',y = 'empty', ax=axes[4])
    df_AAPL.reset_index().plot(kind='scatter',x='index',y='EnterprisesValueEBITDARatio',ax=axes[4], color='#2A9D8F') 
    df_MSFT.reset_index().plot(kind='scatter',x='index',y='EnterprisesValueEBITDARatio',ax=axes[4], color='#E9C46A') 
    df_KO.reset_index().plot(kind='scatter',x='index',y='EnterprisesValueEBITDARatio',ax=axes[4], color='#E76F51')
    axes[4].plot([], [], 'o', color='#2A9D8F', label = 'AAPL')
    axes[4].plot([], [], 'o', color='#E9C46A', label = 'MSFT')
    axes[4].plot([], [], 'o', color='#E76F51', label = 'KO')
    axes[4].set_ylabel("EV/EBITDA(-)")

    plt.savefig('./thesis_datasets/images/dataset4_quarterly.png')
    plt.show()

def plot_example_AAPL_features_dataset4():
    FILEPATH = './thesis_datasets/Dataset4/AAPL_Price_and_Quarterly.csv'
    
    df = pd.read_csv(FILEPATH)	
    data_name = 'Close'

    imputer1 = 'linearfit260'

    EnterpriseValue1 ='EnterpriseValue_' + imputer1
    PeRatio1 = 'PeRatio_' + imputer1
    ForwardPeRatio1 = 'ForwardPeRatio_' + imputer1
    PegRatio1 = 'PegRatio_' + imputer1
    EnterprisesValueEBITDARatio1 = 'EnterprisesValueEBITDARatio_' + imputer1

    time_lag = 'time_lag'

    data = df.filter(items=[data_name, EnterpriseValue1, PeRatio1,
     ForwardPeRatio1, PegRatio1, EnterprisesValueEBITDARatio1, time_lag])
    
    scaler = MinMaxScaler(feature_range=(-1, 1))

    data_scaled = scaler.fit_transform(data)
    data = pd.DataFrame(data_scaled, columns=data.columns)

    y_train, y_test = temporal_train_test_split(data[data_name], test_size=20)

    y_train_EV , y_test_EV = temporal_train_test_split(data[EnterpriseValue1], test_size=20)

    y_train_PE, y_test_PE= temporal_train_test_split(data[PeRatio1], test_size=20)

    y_train_Forward_PE, y_test_Forward_PE = temporal_train_test_split(data[ForwardPeRatio1], test_size=20)

    y_train_PEG, y_test_PEG = temporal_train_test_split(data[PegRatio1], test_size=20)

    y_train_EV_EBITDA , y_test_EV_EBITDA = temporal_train_test_split(data[EnterprisesValueEBITDARatio1], test_size=20)

    y_train_time_lag, y_test_time_lag= temporal_train_test_split(data[time_lag], test_size=20)

    # sns.set(rc={'figure.figsize':(10,5)})

    fig2, ax = plot_series(
    y_train,
    y_train_EV,
    y_train_PE,
    y_train_Forward_PE,
    y_train_PEG,
    y_train_EV_EBITDA,
    y_test, y_train_time_lag,
    labels=["Price-train",'Enterprise Value','P/E-ratio','Forward P/E-ratio',
    'PEG-ratio', 'EV/EBITDA-ratio',"Price-test","time-lag"]
    )

    

    ax.get_lines()[0].set_color("#264653")
    ax.get_lines()[1].set_color("#2A9D8F")
    ax.get_lines()[2].set_color("#E9C46A")
    ax.get_lines()[3].set_color("#F4A261")
    ax.get_lines()[4].set_color("#E76F51")
    ax.get_lines()[5].set_color("#7AAEC2")
    ax.get_lines()[6].set_color("#84DED4")
    ax.get_lines()[7].set_color("#DEEBF0")

    ax.get_lines()[1].set_linewidth(1.2)
    ax.get_lines()[2].set_linewidth(1.2)
    ax.get_lines()[3].set_linewidth(1.2)
    ax.get_lines()[4].set_linewidth(1.2)
    ax.get_lines()[5].set_linewidth(1.2)
    
    ax.get_lines()[0].set_markevery(5)
    ax.get_lines()[1].set_markevery(5)
    ax.get_lines()[2].set_markevery(5)
    ax.get_lines()[3].set_markevery(5)
    ax.get_lines()[4].set_markevery(5)
    ax.get_lines()[5].set_markevery(5)
    ax.get_lines()[6].set_markevery(5)

    ax.get_lines()[7].set_markevery(5)
    

    ax.legend()

    fig2.set_size_inches(12, 6)

    ax.set_title("train- and test-set: AAPL stock price, time-lag and quarterly figures imputed with linearfit260")
    ax.set_ylabel("Scaled feature value between -1 and 1")
    plt.legend(loc='best')
    plt.savefig('./thesis_datasets/images/dataset4_AAPL_example.png')
    plt.show()
    

# plot_dataset1()
# plot_dataset2()
# plot_dataset2_subplots()
# plot_dataset3_subplots()
# plot_dataset4_quarterly()

plot_example_AAPL_features_dataset4()