# y = 2x+3 그래프
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm                
from sklearn.linear_model import LinearRegression

x = np.linspace(0,100,100)
y = 2*x + 3

#np.random.seed(20240805)
obs_x = np.random.choice(np.arange(100),20)
epsilon_i = norm.rvs(loc=0,scale=20,size=20)
obs_y=2*obs_x + 3 + epsilon_i


plt.plot(x,y,color = 'black')
plt.scatter(obs_x,obs_y, color = 'blue', s=20)
#plt.show()
#plt.clf()


# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
obs_x = obs_x.reshape(-1,1)
model.fit(obs_x, obs_y)  # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_[0]      # 기울기 a hat
model.intercept_ # 절편 b hat

# 예측값 계산
y_pred = model.predict(obs_x)

# 데이터와 회귀 직선 시각화
x = np.linspace(0,100,400)
y = model.coef_[0] * x + model.intercept_
plt.xlim([0,100])
plt.ylim([0,300])
plt.plot(x,y,color='red') # 회귀직선
plt.legend()
plt.show()
plt.clf()

# !pip install statsmodels
import statsmodels.api as sm

obs_x = sm.add_constant(obs_x)
model = sm.OLS(obs_y, obs_x).fit()
print(model.summary())



norm.cdf(18,loc=10,scale=1.96)



# p.57 연습문제 (1, 5번 제외)
x = np.array([15.078, 15.752, 15.549, 15.56, 16.098, 13.277, 15.462, 16.116, 15.214, 16.93, 14.118, 14.927,15.382, 16.709, 16.804])

# 2. H0 : 소비효율 >= 16.0, Ha : 소비효율 < 16.0

n = len(x)
x_mean = x.mean()
s = np.std(x, ddof=0)

t = (x_mean - 16) / (s/np.sqrt(n-1))

