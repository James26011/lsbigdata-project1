import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# X~(3,7^2), 하위 25%
x = norm.ppf(0.25,loc=3,scale=7)
x

# Z~N(0,1^2), 하위 25%
z = norm.ppf(0.25,loc=0, scale=1)
z

x
3+ z*7

norm.cdf(5,loc=3,scale=7)
norm.cdf(2/7,loc=0,scale=1)

norm.ppf(0.975,loc=0,scale=1)

===============================================
# 표준정규분포 표본 1,000개 히스토그램,pdf 겹쳐서 그리기
z=norm.rvs(loc=0, scale=1, size=1000)
z

sns.histplot(z, stat="density",color = 'gray')

# Plot the normal distribution PDF
zmin, zmax = (z.min(), z.max())
z_values = np.linspace(zmin, zmax, 100)
pdf_values = norm.pdf(z_values, loc=0, scale=1)
plt.plot(z_values, pdf_values, color='red', linewidth=2)

plt.show()
# plt.clf()


x = 3 + (z*np.sqrt(2))
sns.histplot(x, stat="density",color = 'green')

# Plot the normal distribution PDF
zmin, zmax = (z.min(), x.max())
z_values = np.linspace(zmin, zmax, 100)
pdf_values = norm.pdf(z_values, loc=3, scale=np.sqrt(2))
plt.plot(z_values, pdf_values, color='red', linewidth=2)

plt.show()
plt.clf()

# ======표준화 확인========================================
# 예제 1
x=norm.rvs(loc=5, scale=3, size=1000)
z = (x - 5) / 3
sns.histplot(z, stat="density",color = 'gray')

zmin, zmax = (z.min(), z.max())
z_values = np.linspace(zmin, zmax, 100)
pdf_values = norm.pdf(z_values, loc=0, scale=1)
plt.plot(z_values, pdf_values, color='red', linewidth=2)
plt.show()
plt.clf()

# 예제 2 - 표본표준편차를 나눠도 표준정규분포가 될까?
# 1.
x=norm.rvs(loc=5, scale=3, size=10)
s = np.std(x, ddof =1)
s
# 2.
x=norm.rvs(loc=5, scale=3, size=1000)
z = (x-5)/ s
#z = (x-5) / 3
sns.histplot(z, stat="density",color = 'gray')

zmin, zmax = (z.min(), z.max())
z_values = np.linspace(zmin, zmax, 100)
pdf_values = norm.pdf(z_values, loc=0, scale=1)
plt.plot(z_values, pdf_values, color='red', linewidth=2)
plt.show()
plt.clf()
# 더 이상 표준정규분포가 아니다!






#========t 분포==============
# X ~ t(df) / 종모양, 대칭, 중심 0
# 모수 df : 자유도, df은 분산에 영향 - 퍼짐을 나타내는 모수
# df가 작으면 분산이 커짐
# 자유도가 40정도 되면 표준정규분포 모양이다.

from scipy.stats import t
# t.pdf
# t.ppf
# t.cdf
# t.rvs

# 자유도가 4인 t분포의 pdf를 그리시오
t_values = np.linspace(-4,4, 100)
pdf_values = t.pdf(t_values, df = 50) # df=자유도
plt.plot(t_values, pdf_values, color='red', linewidth=2)
plt.show()
#plt.clf()

z=norm.rvs(loc=0, scale=1, size=1000)
sns.histplot(z, stat="density",color = 'gray')

# Plot the normal distribution PDF
zmin, zmax = (z.min(), z.max())
z_values = np.linspace(zmin, zmax, 100)
pdf_values = norm.pdf(z_values, loc=0, scale=1)
plt.plot(z_values, pdf_values, color='black', linewidth=2)

plt.show()
plt.clf()


# =======================
# X ~ ?(mu, sigma^2)
# X bar ~ N(mu,sigma^2/n)
# X bar ~= t(x_bar,s^2/n) 자유도가 n-1안 t 분포










