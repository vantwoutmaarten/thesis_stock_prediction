#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 5]



df = pd.read_excel ('./Experiment_Excel.xlsx', sheet_name='ImputationBoxplot')
plt.figure()
plt.subplot(2,1,1)
ax = sns.swarmplot(x='imputation method', y='smape', hue='Missingness', data = df[df['stock']=='Apple'])
plt.subplot(2,1,2)
ax = sns.swarmplot(x='imputation method', y='smape', hue='Missingness', data = df[df['stock']=='CocaCola'])

# %%
