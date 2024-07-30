import numpy as np
# 벡터 슬라이싱 예제, a를 랜덤하게 채움
np.random.seed(42)
a = np.random.randint(1, 21, 10)

# 중복 없이 뽑기
?np.random.randint
a = np.random.choice(np.arange(1,21),10,False)
a

print(a)
# 두 번째 값 추출
print(a[1])

a[2:5]
a[-1]
a[::2]

# 1에서부터 1,000사이 3의 배수의 합은?
a = np.arange(0,1000,3)
a
sum(a)

print(a[[0,2,4]])

np.delete(a,[1,3])

a = np.arange(0,100)
a > 3
a[a>3]


np.random.seed(2024)
a = np.random.randint(1,10000,5)
a
a[(a>2000)&(a<5000)]


a>2000
a<5000
(a>2000) & (a<5000)

!pip install pydataset
import pydataset

df = pydataset.data('mtcars')
df
np_df = np.array(df['mpg'])
np_df
model_names = np.array(df.index)
model_names

#연비가 15 이상 25이하인 데이터 개수?

sum((np_df >= 15) & (np_df <= 25))

# 평균 mpg보다 높은(이상) 자동차 대수는?
sum(np_df >= np.mean(np_df))

# 15 작거나 22이상인 데이터 개수는?
sum((np_df < 15) | (np_df >= 22))

#15 이상 20이하인 자동차 모델은?
model_names[(np_df >= 15) & (np_df <=20)]

# 평균mpg보다 낮은(미만) 자동차 모델은?
model_names[np_df < np.mean(np_df)]

df['mpg'][df['mpg']>30]

np.random.seed(2024)
a = np.random.randint(1,10000,5)
b = np.array(['A','B','C','F','W'])
a
b
a[(a>2000) & (a<5000)]
b[(a>2000) & (a<5000)]


np.random.seed(2024)
a = np.random.randint(1,100,10)
a
np.where(a<50) #True인 인덱스

np.random.seed(2024)
a = np.random.randint(1,26346,1000)
a
#처음으로 5000보다 큰 숫자가 나오는 숫자?
a[a>5000][0]
a[np.where(a>5000)]

#처음으로 2200보다 큰 숫자가 나왔을 때, 숫자 위치와 그 숫자는?
x = np.where(a>22000)
type(x)
x
x[0][0]
#0번째 인덱스에 np어레이가 들어있고 튜플 형태이다.0번째의 0번째를 가져와야한다.
a[np.where(a > 22000)][0]

#처음으로 10,000보다 큰 숫자가 나왔을 때, 50번째 나오는 숫자와 위치?
x = np.where(a > 10000)
x
x[0][49]
a[np.where(a>10000)][81]

#500보다 작은 숫자들 중 가장 마지막으로 나오는 숫자 위치와 그 숫자?
x = np.where(a < 500)
x
a[x[0][-1]] #960번째, 391


a = np.array([20, np.nan, 13, 24, 309])
np.isnan(a)
a
a + 3
np.mean(a)
np.nanmean(a)
np.nan_to_num(a, nan = 0)


a = None
a #찍으면 없는 상태

a + 1

a = np.array([20, np.nan, 13, 24, 309])
a_filtered = a[~np.isnan(a)]
a_filtered

str_vec = np.array(["사과", "배", "수박", "참외"])
str_vec
str_vec[[0, 2]]

mix_vec = np.array(["사과",12, "배", "수박", "참외"], dtype = str) 
#리스트 안에 데이터 형식이 달라도 된다
mix_vec

combined_vec = np.concatenate((str_vec,mix_vec))
combined_vec


col_stacked = np.column_stack((np.arange(1, 5), np.arange(12, 16)))
col_stacked

row_stacked = np.row_stack((np.arange(1, 5), np.arange(12, 16)))
row_stacked


uneven_stacked = np.column_stack((np.arange(1, 5), np.arange(12, 18)))
uneven_stacked

# 길이가 다른 벡터
vec1 = np.arange(1, 5)
vec2 = np.arange(12, 18)
vec1 = np.resize(vec1, len(vec2))
vec1

uneven_stacked = np.column_stack((vec1,vec2))
uneven_stacked

#연습문제1
a = np.array([1, 2, 3, 4, 5])
a + 5

#연습문제2
a = np.array([12, 21, 35, 48, 5])
a[0::2]
a[a%2==1]

#연습문제3
a = np.array([1, 22, 93, 64, 54])
a.max()

#연습문제4
a = np.array([1, 2, 3, 2, 4, 5, 4, 6])
np.unique(a)

#연습문제5
a = np.array([21, 31, 58])
b = np.array([24, 44, 67])

x = np.empty(6)
# 짝수
x[1::2] = a
#홀수
x[0::2] = b
x
