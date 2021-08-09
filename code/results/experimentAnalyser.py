#%%
from typing import DefaultDict
from matplotlib import colors
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from seaborn import palettes
from seaborn.miscplot import palplot
plt.rcParams['figure.figsize'] = [10, 5]
import xlsxwriter


def baseline_swarms():
    print('BASELINE SWARMS')
    df = pd.read_excel ('./results/baseline/BaselineRaw.xlsx', sheet_name='BaselineRaw')

    plt.rcParams['figure.figsize'] = [5, 4]
    fig = plt.figure()
    fig.tight_layout()
    fig.subplots_adjust(left=0.16)
    palette = ["#2A9D8F", "#E76F51", "#E9C46A"]
    ax = sns.swarmplot(x='company', y='smape', hue='company', data = df.sort_values(by=['company']), palette=palette).set_title('Baseline forecasting performance of companies')
    print('BASELINE SWARMS')
    
    plt.savefig('./results/baseline/baseline_Combination.png')
    plt.show()


def baseline_error_tables():
    print('BASELINE TABLES')
    df = pd.read_excel ('./results/baseline/BaselineRaw.xlsx', sheet_name='BaselineRaw')

    results_mean = df.groupby(['company']).smape.mean()
    results_std = df.groupby(['company']).smape.std()

    results_processed = pd.concat([results_mean, results_std], axis=1) 
    results_processed.columns= ['smape mean', 'smape std']

    with pd.ExcelWriter('./results/baseline/BaselineTable.xlsx', engine='xlsxwriter') as writer:   
        results_processed.to_excel(writer, sheet_name='BaselineTable')

def experiment1_swarms():
    print('Experiment 1 SWARMS')
    df = pd.read_excel ('./results/experiment1/Experiment-1Raw.xlsx', sheet_name='Experiment-1Raw')

    # a plot with all companies together. 
    plt.rcParams['figure.figsize'] = [10, 4]
    fig = plt.figure()
    fig.tight_layout()
    palette = ["#2A9D8F", "#E76F51", "#E9C46A"]
    ax = sns.swarmplot(x='imputation', y='smape', hue='company', data = df.sort_values(by=['company','imputation']), palette=palette).set_title('Forecasting performance of three companies with 98.5 percent missingness')
    plt.savefig('./results/experiment1/experiment1.png')
    plt.show()

def experiment1_error_tables():
    print('Experiment 1 error table')
    df = pd.read_excel ('./results/experiment1/Experiment-1Raw.xlsx', sheet_name='Experiment-1Raw')

    results = df

    results_mean = results.groupby(['company','imputation']).smape.mean()
    results_std = results.groupby(['company','imputation']).smape.std()

    results_mean = results_mean.rename('smape mean')
    results_std = results_std.rename('smape std')
    results_processed_splitbycompany = pd.concat([results_mean, results_std], axis=1)


    with pd.ExcelWriter('./results/experiment1/Experiment1-Table.xlsx', engine='xlsxwriter') as writer:   
        results_processed_splitbycompany.to_excel(writer, sheet_name='Experiment1-Table')

def experiment2_swarms():
    print('Experiment 2 SWARMS')
    df = pd.read_excel ('./results/experiment2/Experiment-2RAW.xlsx', sheet_name='Experiment-2RAW')

    plt.rcParams['figure.figsize'] = [10, 4]
    fig = plt.figure()
    fig.tight_layout()

    plt.title("Apple with shifted ahead version with different missingness")
    palette = ["#264653", "#2A9D8F", "#E76F51"]
    ax = sns.swarmplot(x='imputation', y='smape', hue='missingness', data = df[df['company']=='AAPL'].sort_values(by=['imputation']), palette=palette)
    plt.savefig('./results/experiment2/experiment2-AAPL.png')
    plt.show()
    
    fig = plt.figure()
    fig.tight_layout()
    plt.title("CocaCola with shifted ahead version with different missingness")
    palette = ["#264653", "#2A9D8F", "#E76F51"]
    ax = sns.swarmplot(x='imputation', y='smape', hue='missingness', data = df[df['company']=='KO'].sort_values(by=['imputation']), palette=palette)
    plt.savefig('./results/experiment2/experiment2-KO.png')
    plt.show()

    fig = plt.figure()
    fig.tight_layout()
    plt.title("Microsoft with shifted ahead version with different missingness")
    palette = ["#264653", "#2A9D8F", "#E76F51"]
    ax = sns.swarmplot(x='imputation', y='smape', hue='missingness', data = df[df['company']=='MSFT'].sort_values(by=['imputation']), palette=palette)
    plt.savefig('./results/experiment2/experiment2-MSFT.png')
    plt.show()

def experiment2_error_tables():
    print('Experiment 2 error table')
    df = pd.read_excel ('./results/experiment2/Experiment-2RAW.xlsx', sheet_name='Experiment-2RAW')

    allresults = df

    # Create table for missigness 33% with two indexes company and imputation method.
    results_missing33 = allresults[allresults['missingness']==33]

    results_missing33_mean = results_missing33.groupby(['company','imputation']).smape.mean()
    results_missing33_std = results_missing33.groupby(['company','imputation']).smape.std()

    results_missing33_mean = results_missing33_mean.rename('smape_mean')
    results_missing33_std = results_missing33_std.rename('smape_std')

    results_missing33_processed_splitbycompany = pd.concat([results_missing33_mean, results_missing33_std], axis=1) 

    # Create table for missigness 90% with two indexes company and imputation method.
    results_missing90 = allresults[allresults['missingness']==90]

    results_missing90_mean = results_missing90.groupby(['company','imputation']).smape.mean()
    results_missing90_std = results_missing90.groupby(['company','imputation']).smape.std()

    results_missing90_mean = results_missing90_mean.rename('smape_mean')
    results_missing90_std = results_missing90_std.rename('smape_std')

    results_missing90_processed_splitbycompany = pd.concat([results_missing90_mean, results_missing90_std], axis=1) 

    # Create table for missigness 98.5% with two indexes company and imputation method.
    results_missing985 = allresults[allresults['missingness']==985]

    results_missing985_mean = results_missing985.groupby(['company','imputation']).smape.mean()
    results_missing985_std = results_missing985.groupby(['company','imputation']).smape.std()

    results_missing985_mean = results_missing985_mean.rename('smape_mean')
    results_missing985_std = results_missing985_std.rename('smape_std')

    results_missing985_processed_splitbycompany = pd.concat([results_missing985_mean, results_missing985_std], axis=1) 


    # Now I will write the 3 tables to 3 sheets and then combine them manually.
    #Quickly check what happens when written to same sheet. 

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    with pd.ExcelWriter('./results/experiment2/Experiment2-Table.xlsx', engine='xlsxwriter') as writer:    
        # Write each dataframe to a different worksheet.
        results_missing33_processed_splitbycompany.to_excel(writer, sheet_name='missingness33_2indexes')
        results_missing90_processed_splitbycompany.to_excel(writer, sheet_name='missingness90_2indexes')
        results_missing985_processed_splitbycompany.to_excel(writer, sheet_name='missingness985_2indexes')


def experiment3and5swarms():
    print('Experiment 3 SWARMS')
    df3 = pd.read_excel ('./results/experiment3/Experiment-3RAW.xlsx', sheet_name='Experiment-3RAW')
    df3['feature'] = 'no-extra_feature'

    df5 = pd.read_excel ('./results/experiment5/Experiment-5RAW.xlsx', sheet_name='Experiment-5RAW')

    df5_time_lag = df5[df5['Tags'].apply(lambda x: 'time-lag' in x)]
    df5_time_lag['feature'] = 'time-lag'

    df5_presence = df5[df5['Tags'].apply(lambda x: 'presence' in x)]
    df5_presence['feature'] = 'missing-1or1'

    frames = [df3, df5_time_lag, df5_presence]

    df = pd.concat(frames)
    print(df.head())

    plt.rcParams['figure.figsize'] = [10, 4]
    fig = plt.figure()
    fig.tight_layout()
    plt.title("Apple with quarterly figures with additional missiness feature")
    palette = ["#264653", "#2A9D8F", "#E76F51"]
    ax = sns.swarmplot(x='imputation', y='smape', hue='feature', data = df[df['company']=='AAPL'].sort_values(by=['imputation', 'feature']), palette=palette)
    plt.savefig('./results/experiment3/experiment3-AAPL.png')
    plt.show()
    
    fig = plt.figure()
    fig.tight_layout()

    plt.title("CocaCola with quarterly figures with additional missiness feature")
    palette = ["#264653", "#2A9D8F", "#E76F51"]
    ax = sns.swarmplot(x='imputation', y='smape', hue='feature', data = df[df['company']=='KO'].sort_values(by=['imputation', 'feature']), palette=palette)
    plt.savefig('./results/experiment3/experiment3-KO.png')
    plt.show()

    fig = plt.figure()
    fig.tight_layout()

    plt.title("Microsoft with quarterly figures with additional missiness feature")
    palette = ["#264653", "#2A9D8F", "#E76F51"]
    ax = sns.swarmplot(x='imputation', y='smape', hue='feature', data = df[df['company']=='MSFT'].sort_values(by=['imputation', 'feature']), palette=palette)
    plt.savefig('./results/experiment3/experiment3-MSFT.png')
    plt.show()

def experiment3and5_error_tables():
    print('Experiment 3 SWARMS')
    df3 = pd.read_excel ('./results/experiment3/Experiment-3RAW.xlsx', sheet_name='Experiment-3RAW')
    df3['feature'] = 'no-extra_feature'

    df5 = pd.read_excel ('./results/experiment5/Experiment-5RAW.xlsx', sheet_name='Experiment-5RAW')

    df5_time_lag = df5[df5['Tags'].apply(lambda x: 'time-lag' in x)]
    df5_time_lag['feature'] = 'time-lag'

    df5_presence = df5[df5['Tags'].apply(lambda x: 'presence' in x)]
    df5_presence['feature'] = 'missing-1or1'

    frames = [df3, df5_time_lag, df5_presence]

    allresults = pd.concat(frames)

    # Create table for missigness 33% with two indexes company and imputation method.
    results_no_extra_feature = allresults[allresults['feature']=='no-extra_feature']

    results_no_extra_feature_mean = results_no_extra_feature.groupby(['company','imputation']).smape.mean()
    results_no_extra_feature_std = results_no_extra_feature.groupby(['company','imputation']).smape.std()

    results_no_extra_feature_mean = results_no_extra_feature_mean.rename('smape_mean')
    results_no_extra_feature_std = results_no_extra_feature_std.rename('smape_std')

    results_no_extra_feature_processed_splitbycompany = pd.concat([results_no_extra_feature_mean, results_no_extra_feature_std], axis=1) 

    # Create table for missigness 90% with two indexes company and imputation method.
    results_time_lag = allresults[allresults['feature']=='time-lag']

    results_time_lag_mean = results_time_lag.groupby(['company','imputation']).smape.mean()
    results_time_lag_std = results_time_lag.groupby(['company','imputation']).smape.std()

    results_time_lag_mean = results_time_lag_mean.rename('smape_mean')
    results_time_lag_std = results_time_lag_std.rename('smape_std')

    results_time_lag_processed_splitbycompany = pd.concat([results_time_lag_mean, results_time_lag_std], axis=1) 

    # Create table for missigness 98.5% with two indexes company and imputation method.
    results_presence = allresults[allresults['feature']=='missing-1or1']

    results_presence_mean = results_presence.groupby(['company','imputation']).smape.mean()
    results_presence_std = results_presence.groupby(['company','imputation']).smape.std()

    results_presence_mean = results_presence_mean.rename('smape_mean')
    results_presence_std = results_presence_std.rename('smape_std')

    results_presence_processed_splitbycompany = pd.concat([results_presence_mean, results_presence_std], axis=1) 


    # Now I will write the 3 tables to 3 sheets and then combine them manually.
    #Quickly check what happens when written to same sheet. 

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    with pd.ExcelWriter('./results/experiment3/Experiment3-Tables.xlsx', engine='xlsxwriter') as writer:    
        # Write each dataframe to a different worksheet.
        results_no_extra_feature_processed_splitbycompany.to_excel(writer, sheet_name='no-extra_feature')
        results_time_lag_processed_splitbycompany.to_excel(writer, sheet_name='time-lag')
        results_presence_processed_splitbycompany.to_excel(writer, sheet_name='missing-1or1')

def experiment4_swarms():
    print('Experiment 4 SWARMS')
    df2 = pd.read_excel('./results/experiment2/Experiment-2RAW.xlsx', sheet_name='Experiment-2RAW')

    df2['feature'] = 'no-extra_feature'

    df4 = pd.read_excel ('./results/experiment4/Experiment-4RAW-v3.xlsx', sheet_name='Experiment-4RAW-v3')

    df4_time_lag = df4[df4['Tags'].apply(lambda x: 'time-lag' in x)]
    df4_time_lag['feature'] = 'time-lag'

    df4_presence = df4[df4['Tags'].apply(lambda x: 'presence' in x)]
    df4_presence['feature'] = 'missing-1or1'

    frames = [df2, df4_time_lag, df4_presence]

    df = pd.concat(frames)
    
    # Here missingness is specified, the others are also experimented with! Lets decide later whether to include that, depends on the results. 
    df = df[df['missingness']==985]

    print(df.head())
    
    plt.rcParams['figure.figsize'] = [10, 4]
    fig = plt.figure()
    fig.tight_layout()

    plt.title("Apple with shifted ahead version 98.5% missing with additional missiness feature")
    palette = ["#264653", "#2A9D8F", "#E76F51"]
    ax = sns.swarmplot(x='imputation', y='smape', hue='feature', data = df[df['company']=='AAPL'].sort_values(by=['imputation', 'feature']), palette=palette)
    plt.savefig('./results/experiment4/experiment4-AAPL.png')
    plt.show()
    
    fig = plt.figure()
    fig.tight_layout()
    plt.title("CocaCola with shifted ahead version 98.5% missing with additional missiness feature")
    palette = ["#264653", "#2A9D8F", "#E76F51"]
    ax = sns.swarmplot(x='imputation', y='smape', hue='feature', data = df[df['company']=='KO'].sort_values(by=['imputation', 'feature']), palette=palette)
    plt.savefig('./results/experiment4/experiment4-KO.png')
    plt.show()

    fig = plt.figure()
    fig.tight_layout()
    plt.title("Microsoft with shifted ahead version 98.5% missing with additional missiness feature")
    palette = ["#264653", "#2A9D8F", "#E76F51"]
    ax = sns.swarmplot(x='imputation', y='smape', hue='feature', data = df[df['company']=='MSFT'].sort_values(by=['imputation', 'feature']), palette=palette)
    plt.savefig('./results/experiment4/experiment4-MSFT.png')
    plt.show()

def experiment4_error_tables():
    print('Experiment 4 SWARMS')
    df2 = pd.read_excel('./results/experiment2/Experiment-2RAW.xlsx', sheet_name='Experiment-2RAW')

    df2['feature'] = 'no-extra_feature'

    df4 = pd.read_excel ('./results/experiment4/Experiment-4RAW-v3.xlsx', sheet_name='Experiment-4RAW-v3')

    df4_time_lag = df4[df4['Tags'].apply(lambda x: 'time-lag' in x)]
    df4_time_lag['feature'] = 'time-lag'

    df4_presence = df4[df4['Tags'].apply(lambda x: 'presence' in x)]
    df4_presence['feature'] = 'missing-1or1'

    frames = [df2, df4_time_lag, df4_presence]

    df = pd.concat(frames)
    
    # Here missingness is specified, the others are also experimented with! Lets decide later whether to include that, depends on the results. 
    allresults = df[df['missingness']==985]

    # Create table for missigness 33% with two indexes company and imputation method.
    results_no_extra_feature = allresults[allresults['feature']=='no-extra_feature']

    results_no_extra_feature_mean = results_no_extra_feature.groupby(['company','imputation']).smape.mean()
    results_no_extra_feature_std = results_no_extra_feature.groupby(['company','imputation']).smape.std()

    results_no_extra_feature_mean = results_no_extra_feature_mean.rename('smape_mean')
    results_no_extra_feature_std = results_no_extra_feature_std.rename('smape_std')

    results_no_extra_feature_processed_splitbycompany = pd.concat([results_no_extra_feature_mean, results_no_extra_feature_std], axis=1) 

    # Create table for missigness 90% with two indexes company and imputation method.
    results_time_lag = allresults[allresults['feature']=='time-lag']

    results_time_lag_mean = results_time_lag.groupby(['company','imputation']).smape.mean()
    results_time_lag_std = results_time_lag.groupby(['company','imputation']).smape.std()

    results_time_lag_mean = results_time_lag_mean.rename('smape_mean')
    results_time_lag_std = results_time_lag_std.rename('smape_std')

    results_time_lag_processed_splitbycompany = pd.concat([results_time_lag_mean, results_time_lag_std], axis=1) 

    # Create table for missigness 98.5% with two indexes company and imputation method.
    results_presence = allresults[allresults['feature']=='missing-1or1']

    results_presence_mean = results_presence.groupby(['company','imputation']).smape.mean()
    results_presence_std = results_presence.groupby(['company','imputation']).smape.std()

    results_presence_mean = results_presence_mean.rename('smape_mean')
    results_presence_std = results_presence_std.rename('smape_std')

    results_presence_processed_splitbycompany = pd.concat([results_presence_mean, results_presence_std], axis=1) 


    # Now I will write the 3 tables to 3 sheets and then combine them manually.
    #Quickly check what happens when written to same sheet. 

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    with pd.ExcelWriter('./results/experiment4/Experiment4-Tables.xlsx', engine='xlsxwriter') as writer:    
        # Write each dataframe to a different worksheet.
        results_no_extra_feature_processed_splitbycompany.to_excel(writer, sheet_name='no-extra_feature')
        results_time_lag_processed_splitbycompany.to_excel(writer, sheet_name='time-lag')
        results_presence_processed_splitbycompany.to_excel(writer, sheet_name='missing-1or1')

#Baseline 1
# baseline_swarms()

#error table is now created
# baseline_error_tables()

#Experiment 1
# experiment1_swarms()
# experiment1_error_tables()

#Experiment 2
# experiment2_swarms()
# experiment2_error_tables()


#Experiment 3
experiment3and5swarms()
# experiment3and5_error_tables()

#Experiment 4
experiment4_swarms()
experiment4_error_tables()

