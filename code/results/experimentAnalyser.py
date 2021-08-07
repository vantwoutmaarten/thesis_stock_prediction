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


def plotSwarms():
    plt.figure()
    plt.title("Apple")
    ax = sns.swarmplot(x='imputation', y='smape', hue='missingness', data = df[df['company']=='Apple'].sort_values(by=['imputation']))
    # plt.savefig('./results/baseline/baseline_AAPL.png')

    plt.figure()
    plt.title("CocaCola")
    ax = sns.swarmplot(x='imputation', y='smape', hue='missingness', data = df[df['company']=='CocaCola'].sort_values(by=['imputation']))
    # plt.savefig('./results/baseline/baseline_KO.png')

    plt.figure()
    plt.title("Microsoft")
    ax = sns.swarmplot(x='imputation', y='smape', hue='missingness', data = df[df['company']=='Microsoft'].sort_values(by=['imputation']))



    # plt.savefig('./results/baseline/baseline_MSFT.png')

    #%%
    # plt.figure()
    # g = sns.catplot(x="imputation", y="smape",
    #                 hue="missingness", row="company",
    #                 data=df, kind="swarm");

    # a plot with all companies together. 
    plt.rcParams['figure.figsize'] = [10, 5]
    plt.figure()
    ax = sns.swarmplot(x='imputation', y='smape', hue='missingness', data = df.sort_values(by=['imputation'])).set_title('performance of three companies')
    # plt.savefig('./results/baseline/baseline_Combination.png')

def createErrorTables():

    allresults = df

    # Create table for missigness 33% with two indexes company and imputation method.
    results_missing33 = allresults[allresults['missingness']==33]

    results_missing33_mean = results_missing33.groupby(['company','imputation']).smape.mean()
    results_missing33_std = results_missing33.groupby(['company','imputation']).smape.std()

    results_missing33_mean = results_missing33_mean.rename('smape_mean')
    results_missing33_std = results_missing33_std.rename('smape_std')

    results_missing33_processed_splitbycompany = pd. concat([results_missing33_mean, results_missing33_std], axis=1) 


    # Create table for missigness 33% with one index, imputation method.
    results_missing33 = allresults[allresults['missingness']==33]

    results_missing33_mean = results_missing33.groupby(['imputation']).smape.mean()
    results_missing33_std = results_missing33.groupby(['imputation']).smape.std()

    results_missing33_mean = results_missing33_mean.rename('smape_mean')
    results_missing33_std = results_missing33_std.rename('smape_std')

    results_missing33_processed = pd. concat([results_missing33_mean, results_missing33_std], axis=1) 



    # Create table for missigness 33% with two indexes company and imputation method.
    results_missing90 = allresults[allresults['missingness']==90]

    results_missing90_mean = results_missing90.groupby(['company','imputation']).smape.mean()
    results_missing90_std = results_missing90.groupby(['company','imputation']).smape.std()

    results_missing90_mean = results_missing90_mean.rename('smape_mean')
    results_missing90_std = results_missing90_std.rename('smape_std')

    results_missing90_processed_splitbycompany = pd. concat([results_missing90_mean, results_missing90_std], axis=1) 

    # Create table for missigness 90% with one index, imputation method.
    results_missing90 = allresults[allresults['missingness']==90]

    results_missing90_mean = results_missing90.groupby(['imputation']).smape.mean()
    results_missing90_std = results_missing90.groupby(['imputation']).smape.std()

    results_missing90_mean = results_missing90_mean.rename('smape_mean')
    results_missing90_std = results_missing90_std.rename('smape_std')

    results_missing90_processed = pd. concat([results_missing90_mean, results_missing90_std], axis=1) 

    # Now I will write the four tables to 4 sheets and then combine them manually.
    #Quickly check what happens when written to same sheet. 

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    with pd.ExcelWriter('./impute_excl_imputations_stockprices_test2.xlsx', engine='xlsxwriter') as writer:    
        # Write each dataframe to a different worksheet.
        
        results_missing33_processed.to_excel(writer, sheet_name='missingness33_1index')
        results_missing33_processed_splitbycompany.to_excel(writer, sheet_name='missingness33_2indexes')
        results_missing90_processed.to_excel(writer, sheet_name='missingness90_1index')
        results_missing90_processed_splitbycompany.to_excel(writer, sheet_name='missingness90_2indexes')


def baseline_swarms():
    print('BASELINE SWARMS')
    df = pd.read_excel ('./results/baseline/BaselineRaw.xlsx', sheet_name='BaselineRaw')

    plt.rcParams['figure.figsize'] = [10, 5]
    plt.figure()
    palette = ["#2A9D8F", "#E76F51", "#E9C46A"]
    ax = sns.swarmplot(x='company', y='smape', hue='company', data = df.sort_values(by=['company']), palette=palette).set_title('Baseline forecast performance of companies')
    print('BASELINE SWARMS')
    plt.show()
    
    # plt.savefig('./results/baseline/baseline_Combination.png')

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
    plt.rcParams['figure.figsize'] = [10, 5]
    plt.figure()
    palette = ["#2A9D8F", "#E76F51", "#E9C46A"]
    ax = sns.swarmplot(x='imputation', y='smape', hue='company', data = df.sort_values(by=['company','imputation']), palette=palette).set_title('Forecast performance of three companies with 98.5 percent missingness')
    # plt.savefig('./results/baseline/baseline_Combination.png')
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


#Baseline 1
#still have to save image, but ready
# baseline_swarms()

#error table is now created
# baseline_error_tables()

#Experiment 1, last subexperiments still missing
#still have to save image, but ready
experiment1_swarms()
# experiment1_error_tables()


