from scipy.stats import uniform
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.stats import norm
import math
import numpy as np

uniform.rvs(loc=2, scale=4,size=1)

# uniform.pdf(x, loc=0, scale=1)
# uniform.cdf(x, loc=0, scale=1)
# uniform.ppf(q, loc=0, scale=1)
uniform.pdf(k, loc=2, scale=4)
uniform.rvs(loc=0, scale=1, size=None, random_state=None)


k = np.linspace(0,8,100)
y = uniform.pdf(k, loc=2, scale=4)
plt.plot(k,y,color='black')
plt.show()
plt.clf()


uniform.cdf(8.39,2,4) - uniform.cdf(5, 2, 4)
uniform.ppf(0.93,2,4)


#표본 20개 뽑고 표본 평균 구하기
x = uniform.rvs(loc=2,scale=4,size=20*1000, random_state=42)
x.shape
x = x.reshape(1000,20)
# x.reshape(-1,20)
blue_x = x.mean(axis=1)
blue_x

sns.histplot(blue_x, stat='density')
plt.show()

# X bar ~ N(mu,sigma^2/n)
# X bar ~ N(4,1.333333/20)
uniform.var(loc=2,scale=4)
uniform.expect(loc=2,scale=4)

xmin,xmax = (blue_x.min(),blue_x.max())
x_values = np.linspace(xmin,xmax,100)
pdf_values = norm.pdf(x_values, loc = 4, scale = np.sqrt(1.3333/20))
plt.plot(x_values,pdf_values, color = 'red', linewidth = 2)
plt.show()
plt.clf()


#신뢰구간 95%
# X bar ~ N(mu,sigma^2/n)
# X bar ~ N(4,1.333333/20)

x_values = np.linspace(3,5,100)
pdf_values = norm.pdf(x_values,loc=4, scale = np.sqrt(1.3333/20))
plt.plot(x_values,pdf_values, color = 'red', linewidth = 2)

plt.show()
plt.clf()
4 - norm.ppf(0.025,loc=4,scale = np.sqrt(1.3333/20))
4 - norm.ppf(0.975,loc=4,scale = np.sqrt(1.3333/20))

x = uniform.rvs(loc=2,scale=4,size=20*1000, random_state=42)
x = x.reshape(1000,20)
x.shape



# 신뢰구간 99%
# X bar ~ N(mu,sigma^2/n)
# X bar ~ N(4,1.333333/20)

x_values = np.linspace(3,5,100)
pdf_values = norm.pdf(x_values,loc=4, scale = np.sqrt(1.3333/20))
plt.plot(x_values,pdf_values, color = 'red', linewidth = 2)

# 기댓값 표현
plt.axvline(x=4,color='green',linestyle = '-',linewidth=2)
# 포인트 추가, 파란 벽돌 점 찍기
blue_x = uniform.rvs(loc=2,scale=4,size=20).mean()
a = blue_x + 0.665
b = blue_x - 0.665
plt.scatter(blue_x, 0.002,color='blue',zorder=10,s=10)
plt. axvline(x=a,color='blue',linestyle='--',linewidth=1)
plt. axvline(x=b,color='blue',linestyle='--',linewidth=1)

plt.show()
plt.clf()
#=====================================================

4 - norm.ppf(0.025,loc=4,scale = np.sqrt(1.3333/20))
4 - norm.ppf(0.975,loc=4,scale = np.sqrt(1.3333/20))

x = uniform.rvs(loc=2,scale=4,size=20*1000, random_state=42)
x = x.reshape(1000,20)
x.shape



