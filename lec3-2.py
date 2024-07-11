# 교재 63p, 패키지 설치

import seaborn as sns
import matplotlib.pyplot as plt

var = ['a','a','b','c']
sns.countplot(x = var)
plt.show()
plt.clf()

df = sns.load_dataset('titanic')
df
sns.countplot(data = df, x = 'sex', hue = 'sex')
plt.show()

df = sns.load_dataset('titanic')
df
sns.countplot(data = df, y = 'class', hue = 'alive')
plt.show()
plt.clf()

sns.countplot(data = df, x = 'class', hue = 'alive', orient='v')
plt.show()
plt.clf()

import sklearn.metrics

from sklearn import metrics


