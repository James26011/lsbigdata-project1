import numpy as np
import matplotlib.pyplot as plt

# 예제 넘파이 배열 생성
data = np.random.rand(1000)

# 히스토그램 그리기
plt.clf()
plt.hist(data, bins = 30, alpha = 0.7, color = 'blue')
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Fre')
plt.grid(True)
plt.show()


# 정규분포도 그리기
plt.clf()
plt.hist(Y(10000), bins = 10000, alpha = 0.7, color = 'blue')
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Fre')
plt.grid(True)
plt.show()

y = np.random.rand(50000) \
             .reshape(-1,5) \
             .mean(axis=1)

x = np.random.rand(99999, 5).mean(axis=1)

plt.clf()
plt.hist(x, bins = 1000, alpha = 0.7, color = 'blue')
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Fre')
plt.grid(True)
plt.show()



# 기댓값과 분산
x = np.arange(33)
np.arange(33).sum() / 33

(np.arange(33) - 16) ** 2
np.unique((np.arange(33) - 16) ** 2)

np.unique((np.arange(33) - 16) ** 2) * (2/33)
# 분산
sum(np.unique((np.arange(33) - 16) ** 2) * (2/33))

# E[X^2]
sum(x**2 * (1/33))

# Var(X) = E[X^2] - (E[X])^2
sum(x**2 * (1/33)) - 16**2


# example 1
x = np.arange(4)
# 1/6, 2/6, 2/6, 1/6
# 이것의 분산은?

pro_x = np.array([1/6,2/6,2/6,1/6])
pro_x
#기댓값
Ex = sum(x * pro_x)
Exx = sum(x**2 *pro_x)
#분산
Exx - Ex**2
sum((x - Ex)**2*pro_x)

# example 2
x = np.arange(99)
pro_x = np.concatenate((np.arange(1,51), np.arange(49,0,-1)))
pro_x = pro_x / 2500
#기댓값
Ex = sum(x * pro_x)
Exx = sum(x**2 *pro_x)
#분산
Exx - Ex**2
sum((x - Ex)**2*pro_x)


# example 3
x = np.arange(0,7,2)
pro_x = np.array([1/6,2/6,2/6,1/6])

#기댓값
Ex = sum(x * pro_x)
Exx = sum(x**2 *pro_x)
#분산
Exx - Ex**2
sum((x - Ex)**2*pro_x)



np.sqrt(9.52**2 / 25)

np.sqrt(3.24)
1.8**2


from scipy.stats import bernoulli
#확률 질량 함수 (pmf)
#확률변수가 갖는 값에 해당하는 확률을 저장하고 있는 함수
# B.pmf(k,p)
bernoulli.pmf(1,0.3)
bernoulli.pmf(0,0.3)

from scipy.stats import binom
# P(X = k | n, p)
# n: 베르누이 확률변수 더한 갯수
# p: 1이 나올 확률
# binom.pmf(k, n, p)
binom.pmf(0, 2, 0.3)
binom.pmf(1, 2, 0.3)
binom.pmf(2, 2, 0.3)

# X ~ B(n, p)
result = [binom.pmf(x,30,0.3) for x in range(31)]
result

# =========================================
# nCr
import math
math.comb(54, 26)

np.cumprod(np.arange(1,5))[-1]
1*2*3*4

np.log(24)
sum(np.log(np.arange(1,5)))
# =========================================

math.comb(2,0) * 1 * 0.7**2
math.comb(2,1) * 0.3**1 * 0.7**1
math.comb(2,2) * 0.3**2 * 0.7**0

binom.pmf(0,2,0.3)
binom.pmf(1,2,0.3)
binom.pmf(2,2,0.3)


binom.pmf(4,10,0.36)

sum(binom.pmf(np.arange(5),10,0.36))

sum(binom.pmf(np.arange(3,9),10,0.36))

a = sum(binom.pmf(np.arange(0,4),30,0.2))
b = sum(binom.pmf(np.arange(25,31),30,0.2))
a+b

1-sum(binom.pmf(np.arange(4,25),30,0.2))


# rvs 함수 (random variates sample)
# 표본 추출 함수
# X1 ~ Bernulli(p=0.3)
bernoulli.rvs(0.3)
# X2 ~ Bernulli(p=0.3)
bernoulli.rvs(0.3)

# X ~ B(n=2, p = 0.3)
bernoulli.rvs(0.3) + bernoulli.rvs(0.3)
binom.rvs(2,0.3)
# 위 2개는 같은 뜻
binom.pmf(0,2,0.3) # x가 0인 확률
binom.pmf(1,2,0.3) # x가 1인 확률
binom.pmf(2,2,0.3) # x가 2인 확률


# X ~ B(30,0.26)
# 표본 30개를 뽑기
binom.rvs(n = 30,p = 0.26, size =30)

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

plt.clf()
prob_x = binom.pmf(np.arange(31),n = 30, p = 0.26)
sns.barplot(prob_x)
plt.show()

# p.207 처럼 시각화
x = np.arange(31)
prob_x = binom.pmf(x,n = 30, p = 0.26)

df = pd.DataFrame({'x' : x, 'prob': prob_x})
df

sns.barplot(data = df, x = 'x', y = 'prob')
plt.show()


# cdf: cumulative dist. fuction
# (누적확률분포함수)
#F(X=x) = P(X <= x)
binom.cdf(4, n=30, p=0.26)

binom.cdf(19, n=30, p=0.26) - binom.cdf(13, n=30, p=0.26)




x_1 = binom.rvs(n=30,p=0.26,size =10)
x_1

x = np.arange(31)
prob_x = binom.pmf(x,n = 30, p = 0.26)
sns.barplot(prob_x, color='blue')

plt.scatter(x_1,np.repeat(0.002,10),color='red', zorder=100,s = 10)

plt.axvline(x=7.8, color='green', linestyle='--', linewidth=2)

plt.show()
plt.clf()


# ppf, P(X < ?) = 0.5
binom.ppf(0.5, n=30,p=0.26)
binom.cdf(7, n=30,p=0.26)

binom.ppf(0.7,n=30,p=0.26)
binom.cdf(8,n=30,p=0.26)


1 / np.sqrt(2*math.pi)
from scipy.stats import norm
norm.pdf(0,loc = 0, scale = 1)

norm.pdf(5,loc = 3, scale = 4)

# 정규분포 pdf 그리기
k = np.linspace(-3,3,100)
y = norm.pdf(k,loc = 0, scale = 1)

plt.scatter(k,y,color='red',s=1)
plt.show()
plt.clf()

# 평균은 종모양의 중심 결정
k = np.linspace(-5,5,100)
y = norm.pdf(k,loc = 3, scale = 1)

plt.plot(k,y,color='black')
plt.show()
plt.clf()

# 시그마(표준편차)는 분포의 퍼짐을 결정
# 모수는 분포의 특성을 결정하는 수
k = np.linspace(-5,5,100)
y = norm.pdf(k,loc = 0, scale = 1)
y2 = norm.pdf(k,loc = 0, scale = 2)
y3 = norm.pdf(k,loc = 0, scale = 0.5)

plt.plot(k,y,color='black')
plt.plot(k,y2,color='red')
plt.plot(k,y3,color='blue')
plt.show()
plt.clf()



norm.cdf(0,loc=0,scale=1)
norm.cdf(100,loc=0,scale=1)

# P(-2 < X 0.54)
norm.cdf(0.54,loc=0,scale=1) - norm.cdf(-2,loc=0,scale=1)

# P(x < 1, or x > 3)
1 - (norm.cdf(1,loc=0,scale=1) - norm.cdf(3,loc=0,scale=1))

# X ~ N(3,5^2)
# P(3 < X < 5) = ?, 15.54%
norm.cdf(5,loc=3,scale=5) - norm.cdf(3,loc=3,scale=5)
# 위 확률변수에서 표본 1000개를 뽑기
x = norm.rvs(loc = 3, scale = 5, size = 1000)
sum((x > 3) & (x < 5)) / 1000

# 평균 0, 표준편차 1
# 표본 1000개 뽑아서 0보다 작은 비율 확인
x = norm.rvs(0,1,1000)
sum(x < 0) / 1000
np.mean(x < 0)

x = norm.rvs(3,2,1000)
sns.histplot(x)
plt.show()
plt.clf()



x = norm.rvs(loc=3,scale=2,size=1000)

sns.histplot(x,stat = 'density')
xmin,xmax = (x.min(),x.max())
x_values = np.linspace(xmin,xmax,100)
pdf_values = norm.pdf(x_values, loc = 3, scale = 2)
plt.plot(x_values,pdf_values, color = 'red', linewidth = 2)

plt.show()
plt.clf()

================================================================


