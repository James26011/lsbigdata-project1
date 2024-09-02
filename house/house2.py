import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# y = 2x+3 그래프 그리기
x = np.linspace(-5, 5, 101)
a = 2
b = 2
y = a*x+b

plt.plot(x,y, color = 'blue')
plt.axvline(0, color = 'black')
plt.axhline(0, color = 'black')
plt.xlim(-5,5)
plt.ylim(-5,5)

plt.show()
plt.clf()


house_train = pd.read_csv('house/train.csv')
df = house_train[['BedroomAbvGr','SalePrice']]
df['SalePrice'] = df['SalePrice'] / 1000

a = 53
b = 45

x = np.linspace(0, 5, 100)
y = a*x+b

plt.plot(x,y, color = 'blue')
plt.scatter(x = df['BedroomAbvGr'], y = df['SalePrice'])
plt.show()
plt.clf()


house_train.groupby('BedroomAbvGr').agg(price = ('BedroomAbvGr',lambda x : 53 * x + 45))
house_train_sub


house_train["SalePrice"] = (house_train['BedroomAbvGr'] * 53 + 45) * 1000
# sub 데이터 불러오기
sub_df = pd.read_csv('house/sample_submission.csv')
sub_df

# 집 가격 바꾸기
sub_df['SalePrice'] = house_train['SalePrice']
sub_df

sub_df.to_csv('./house/sub2.csv', index = False)

#========================================================

# 직선 성능 평가
a = 70
b = 10

# y_hat 어떻게 구할까?
y_hat = (a * house_train['BedroomAbvGr'] + b) * 1000
y = house_train['SalePrice']

np.abs(y - y_hat) # 절대 거리
np.sum(np.abs(y - y_hat))


# =======================
# 회귀 분석
# !pip install scikit-learn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# 예시 데이터 (x와 y 벡터)
x = np.array([1, 3, 2, 1, 5]).reshape(-1, 1)  # x 벡터 (특성 벡터는 2차원 배열이어야 합니다)
y = np.array([1, 2, 3, 4, 5])  # y 벡터 (레이블 벡터는 1차원 배열입니다)

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y)  # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

slope = model.coef_[0]
intercept = model.intercept_
print(f"기울기 (slope a): {slope}")
print(f"절편 (intercept b): {intercept}")

# 예측값 계산
y_pred = model.predict(x)

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='data')
plt.plot(x, y_pred, color='red', label='regression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()

# ===========집 값 예측=================================
# 예시 데이터 (x와 y 벡터)
house_train = pd.read_csv('house/train.csv')

# x 벡터 (특성 벡터는 2차원 배열이어야 합니다)
# y 벡터 (레이블 벡터는 1차원 배열입니다)

x = house_train['BedroomAbvGr'].values.reshape(-1,1)
x
# x = np.array(house_train['BedroomAbvGr']).reshape(-1,1)
y = house_train['SalePrice'].values / 1000
y

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y)  # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a = 1천 6백만씩 방 하나가 늘어날때 값이 증가한다.
model.intercept_ # 절편 b

slope = model.coef_[0]
intercept = model.intercept_
print(f"기울기 (slope a): {slope}")
print(f"절편 (intercept b): {intercept}")

# 예측값 계산
y_pred = model.predict(x)

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='data')
plt.plot(x, y_pred, color='red', label='regression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()

# test 불러오기
house_test = pd.read_csv('house/test.csv')

test_x = np.array(house_test['BedroomAbvGr']).reshape(-1,1)

pred_y = model.predict(test_x) # test 셋에 대한 집값

# sub 데이터 불러오기
sub_df = pd.read_csv('house/sample_submission.csv')
sub_df

# 집 가격 바꾸기
sub_df['SalePrice'] = pred_y * 1000
sub_df


# ==============최솟값 찾기==============================

def my_f(x):
    return x**2 + 3

my_f(3)

import numpy as np
from scipy.optimize import minimize 

# 초기 추정값
initial_guess = [0]

# 최소값 찾기
result = minimize(my_f, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)


# z = x^2 + y^2 + 3
def my_f2(x):
    return x[0]**2 + x[1]*2 + 3

my_f2([1,3])

# 초기 추정값
initial_guess = [-10,3]

# 최소값 찾기
result = minimize(my_f2, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)


# f(x,y,z) = (x-1)^2 + (y-2)^2 + (z-4)^2 + 7
def my_f3(x):
    return (x[0]-1)**2 + (x[1]-2)**2 + (x[2]-4)**2 + 7

my_f2([1,2,4])

# 초기 추정값
initial_guess = [1,2,4]

# 최소값 찾기
result = minimize(my_f3, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)



#==================== 원하는 변수로 회귀모델 만들어서 올리기

# 예시 데이터 (x와 y 벡터)
house_train = pd.read_csv('house/train.csv')

# x 벡터 (특성 벡터는 2차원 배열이어야 합니다)
# y 벡터 (레이블 벡터는 1차원 배열입니다)

# 이상치 탐색
house_train.query('GrLivArea <= 4000')
house_train = house_train.query('GrLivArea <= 4000')

x = house_train['GrLivArea'].values.reshape(-1,1)
x
# x = np.array(house_train['BedroomAbvGr']).reshape(-1,1)
y = house_train['SalePrice'].values
y

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y)  # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a = 1천 6백만씩 방 하나가 늘어날때 값이 증가한다.
model.intercept_ # 절편 b

slope = model.coef_[0]
intercept = model.intercept_
print(f"기울기 (slope a): {slope}")
print(f"절편 (intercept b): {intercept}")

# 예측값 계산
y_pred = model.predict(x)

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='data')
plt.plot(x, y_pred, color='red', label='regression')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([0,5000])
plt.ylim([0,800000])
plt.legend()
plt.show()
plt.clf()


# test 불러오기
house_test = pd.read_csv('house/test.csv')

test_x = np.array(house_test['GrLivArea']).reshape(-1,1)

pred_y = model.predict(test_x) # test 셋에 대한 집값

# sub 데이터 불러오기
sub_df = pd.read_csv('house/sample_submission.csv')
sub_df

# 집 가격 바꾸기
sub_df['SalePrice'] = pred_y
sub_df


sub_df.to_csv('./house/sub8.csv', index = False)



# ======= 원하는 변수 2가지 ===============


# 예시 데이터 (x와 y 벡터)
house_train = pd.read_csv('house/train.csv')

# x 벡터 (특성 벡터는 2차원 배열이어야 합니다)
# y 벡터 (레이블 벡터는 1차원 배열입니다)

# 이상치 탐색
house_train.query('GrLivArea <= 4500')
house_train = house_train.query('GrLivArea <= 4500')

#x = np.array(house_train[['GrLivArea','GarageArea']]).reshape(-1,2)
#y = np.array(house_train['SalePrice'])

#[[]] 이렇게 데이터를 부르면 df로 불러서 np.array 와 reshape을 안해도 된다.
x = house_train[['GrLivArea','GarageArea']]
x
y = house_train['SalePrice']
y

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y)  # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

# f(x,y) = ax + by + c 이런 꼴
#def my_houseprice(x,y):
#    return (model.coef_[0]* x) + (model.coef_[1] * y) + model.intercept_

#my_houseprice(house_test['GrLivArea'], house_test['GarageArea'])

# test 불러오기
house_test = pd.read_csv('house/test.csv')

test_x = house_test[['GrLivArea','GarageArea']]
test_x

# 결측치 확인
test_x['GrLivArea'].isna().sum()
test_x['GarageArea'].isna().sum() # 1개의 결측값

test_x = test_x.fillna(house_test['GarageArea'].mean())
pred_y = model.predict(test_x)
pred_y


# sub 데이터 불러오기
sub_df = pd.read_csv('house/sample_submission.csv')
sub_df

# 집 가격 바꾸기
sub_df['SalePrice'] = pred_y
sub_df


sub_df.to_csv('./house/sub8.csv', index = False)



# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='data')
plt.plot(x, y_pred, color='red', label='regression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()


# ==============240805 숫자형 데이터 모두 활용==========


# 예시 데이터 (x와 y 벡터)
house_train = pd.read_csv('house/train.csv')

# x 벡터 (특성 벡터는 2차원 배열이어야 합니다)
# y 벡터 (레이블 벡터는 1차원 배열입니다)


x = house_train.select_dtypes(include=[int, float])
x.info() 
x = x.iloc[:,1:-1] #id 칼럼, 집가격 칼럼을 제외하자
x.isna().sum() # 3개 변수가 na 값을 가짐
x['LotFrontage'] = x['LotFrontage'].fillna(x['LotFrontage'].mean())
x['MasVnrArea'] = x['MasVnrArea'].fillna(x['MasVnrArea'].mean())
x['GarageYrBlt'] = x['GarageYrBlt'].fillna(x['GarageYrBlt'].mean())
x.isna().sum()


y = house_train['SalePrice']
y

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y)  # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

# f(x,y) = ax + by + c 이런 꼴
#def my_houseprice(x,y):
#    return (model.coef_[0]* x) + (model.coef_[1] * y) + model.intercept_

#my_houseprice(house_test['GrLivArea'], house_test['GarageArea'])

# test 불러오기
house_test = pd.read_csv('house/test.csv')
test_x = house_test[x.columns]


# 결측치 확인
test_x.isna().sum()
test_x = test_x.fillna(test_x.mean())


pred_y = model.predict(test_x)
pred_y


# sub 데이터 불러오기
sub_df = pd.read_csv('house/sample_submission.csv')
sub_df

# 집 가격 바꾸기
sub_df['SalePrice'] = pred_y
sub_df


sub_df.to_csv('./house/sub9.csv', index = True)


