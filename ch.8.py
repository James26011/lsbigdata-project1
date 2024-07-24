import pandas as pd

mpg = pd.read_csv('data/mpg.csv')
mpg.head()
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(data = mpg, x = 'displ', y = 'hwy', hue = 'drv') \
   .set(xlim = [3,6], ylim = [10,30])
plt.show()
plt.clf()

# 막대그래프
mpg['drv'].unique()
df_mpg = mpg.groupby('drv',as_index = False) \
            .agg(mean_hwy = ('hwy','mean'))

df_mpg
plt.clf()
sns.barplot(data = df_mpg, x = 'drv', y = 'mean_hwy', hue = 'drv')
plt.show()


df_mpg = mpg.groupby('drv', as_index = False) \
            .agg(n = ('drv','count'))
df_mpg
plt.clf()
sns.barplot(data = df_mpg, x = 'drv', y = 'n')
plt.show()


