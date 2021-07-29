#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 5]
import xlsxwriter

#%%
# subplots split by company.
df = pd.read_excel ('./excl-imputation-prices.xlsx', sheet_name='excl-imputation-prices')
df.head()


def plotSwarms():
    plt.figure()
    plt.title("Apple")
    ax = sns.swarmplot(x='imputation', y='smape', hue='missingness', data = df[df['company']=='Apple'].sort_values(by=['imputation']))

    plt.figure()
    plt.title("CocaCola")
    ax = sns.swarmplot(x='imputation', y='smape', hue='missingness', data = df[df['company']=='CocaCola'].sort_values(by=['imputation']))

    plt.figure()
    plt.title("Microsoft")
    ax = sns.swarmplot(x='imputation', y='smape', hue='missingness', data = df[df['company']=='Microsoft'].sort_values(by=['imputation']))

    #%%
    # plt.figure()
    # g = sns.catplot(x="imputation", y="smape",
    #                 hue="missingness", row="company",
    #                 data=df, kind="swarm");

    # a plot with all companies together. 
    plt.rcParams['figure.figsize'] = [10, 5]
    plt.figure()
    ax = sns.swarmplot(x='imputation', y='smape', hue='missingness', data = df.sort_values(by=['imputation'])).set_title('performance of three companies')


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


#%%
plotSwarms()

#%%
# createErrorTables()


# %%
