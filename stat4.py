import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 1부터 28까지의 숫자 배열 생성
old_seat = np.arange(1, 29)

# 28까지 중복없이 뽑기
np.random.seed(20240729)
new_seat = np.random.choice(old_seat, 28, replace = False)

result = pd.DataFrame(
    {"old_seat": old_seat,
     "new_seat": new_seat})

# 내보내기
pd.DataFrame.to_csv(result,'result.csv')


# y = 2x 그래프 그리기
x = np.linspace(-10,10,100)
y = 2*x
plt.plot(a,b,color='black')
plt.show()
plt.clf()

# y = x^2를 점 3개 사용해서 그리기
x = np.linspace(-8,8,100)
y = x**2
#plt.scatter(x,y,s=4)
plt.plot(x,y,color='black')
plt.xlim(-10,10)
plt.ylim(0,40)

# plt.axis('equal')은 범위와 같이 사용 ㄴㄴ
plt.gca().set_aspect('equal',adjustable='box')
# 비율 맞추기
#plt.axis('equal')
plt.show()
plt.clf()

# ==============================================
# adp p.39 , 문제 p.57

#작년 남학생 3학년 전체 분포의 표준편차는 6kg 이었다고 합니다. 이 정보를 이번 년도 남학생
#분포의 표준편차로 대체하여 모평균에 대한 90% 신뢰구간을 구하세요.

from scipy.stats import norm
man = np.array([79.1,68.8,62.0,74.4,71.0,60.6,98.5,86.4,73.0,40.8,61.2,68.7,61.6,67.7,61.7,66.8])
man.mean()
len(man)

z_005 = norm.ppf(0.95,loc=0,scale=1)
z_005
# 신뢰구간
man.mean() + z_005 * 6 / np.sqrt(16)
man.mean() - z_005 * 6 / np.sqrt(16)

# 데이터로 부터 E[x^2] 구하기
x = norm.rvs(loc=3,scale=5,size=10000)

np.mean(x**2)
sum(x**2) / (len(x) - 1)

np.mean((x - x**2) / (2*x))

# Q1 표본 분산
np.random.seed(20240729)
x = norm.rvs(loc=3,scale=5,size=100000)
x_bar = x.mean()
s_2 = sum((x-x_bar)**2) / (100000-1)
s_2
np.var(x, ddof = 1) # n-1로 나눈 값 (표본 분산)
np.var(x, ddof = 0) # n으로 나눈 값, 사용 하면 안 됨

# n-1 vs n
x = norm.rvs(loc=3,scale=5,size=20)
np.var(x)
np.var(x, ddof =1)





# 교재 8장, p.212
import seaborn as asns

economics = pd.read_csv('./data/economics.csv')
economics.head()
economics.info()

sns.lineplot(data = economics, x = 'date', y = 'unemploy')
plt.show()
plt.clf()

economics['date2'] = pd.to_datetime(economics['date'])
economics.info()

economics[['date','date2']]
economics['date2']
economics['date2'].dt.year
economics['date2'].dt.month_name()
a = economics['date2'].dt.quarter

economics['quarter'] = a
economics[['date2','quarter']]
economics['date2'].dt.day_name()

economics['date2'] + pd.DateOffset(days=30)
economics['date2'] + pd.DateOffset(months=1)

economics['year'] = economics['date2'].dt.year
economics.head()

sns.lineplot(data = economics, x = 'year', y = 'unemploy', errorbar = None)
plt.show()
plt.clf()
# 적은 표본으로 평균을 내서 모평균을 추정, 


my_df = economics.groupby('year', as_index = False) \
                 .agg(
                    mon_mean = ('unemploy','mean'),
                    mon_std = ('unemploy','std'),
                    mon_n = ('unemploy','count')
                     )
my_df
# mean + 1.96*std/sqrt(12)
my_df['left_ci'] = my_df['mon_mean'] - 1.96 * my_df['mon_std'] / np.sqrt(my_df['mon_n'])
my_df['right_ci'] = my_df['mon_mean'] + 1.96 * my_df['mon_std'] / np.sqrt(my_df['mon_n'])
my_df.head()

x = my_df['year']
y = my_df['mon_mean']

plt.plot(x,y,color = 'black')
plt.scatter(x,my_df['left_ci'],color='blue', s=1)
plt.scatter(x,my_df['right_ci'],color='blue', s=1)
plt.show()
plt.clf()
