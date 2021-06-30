#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15, 10]

#%%
# subplots split by company.
df = pd.read_excel ('./Experiment_Excel.xlsx', sheet_name='ImputationBoxplot')
df.head()

#%%
plt.figure()
plt.title("Apple")
ax = sns.swarmplot(x='imputation method', y='smape', hue='Missingness', data = df[df['stock']=='Apple'])

plt.figure()
plt.title("CocaCola")
ax = sns.swarmplot(x='imputation method', y='smape', hue='Missingness', data = df[df['stock']=='CocaCola'])

plt.figure()
plt.title("Microsoft")
ax = sns.swarmplot(x='imputation method', y='smape', hue='Missingness', data = df[df['stock']=='Microsoft']).set_title('Microsoft')

#%%
plt.figure()
g = sns.catplot(x="imputation method", y="smape",
                hue="Missingness", row="stock",
                data=df, kind="swarm");


#%%
# a plot with all companies together. 
plt.rcParams['figure.figsize'] = [10, 5]
plt.figure()
ax = sns.swarmplot(x='imputation method', y='smape', hue='Missingness', data = df)