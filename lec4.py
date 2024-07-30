# soft copy
a = [1,2,3]
a

b = a
b

a[1] = 4
b
id (a)
id (b)

# deep copy
a = [1,2,3]
a

b=a[:]
b = a.copy()

a[1]=4
a
b

#수학 함수
import math
x = 4
math.sqrt(x) # 제곱근
math.exp(x) #지수 계산
math.log(10,10) #10의 밑 10로그 값
math.factorial(x) # 팩토리얼 계산
math.sin(math.radians(90)) #90도를 라디안으로 변환
math.cos(math.radians(180)) #180도를 라디안으로 변환

# 복잡한 계산
del normal_pdf(x, mu, sigma):
  sqrt_two_pi = math.sqrt(2*math.pi)
  factor = 1 / (sigma * sqrt_two_pi)
  return factor * math.exp(-0.5 * ((x - mu) / sigma) ** 2)



x = 2
y = 9
z = math.pi /2

result = (x**2+math.sqrt(y)+math.sin(z))*math.exp(x)
result

def g(x):
  return math.cos(x) + math.sin(x) * math.exp(x)
g(math.pi)

# ctrl + shift + c = 커맨드 처리
import numpy as np

# 벡터 생성하기 예제
a = np.array([1, 2, 3, 4, 5]) # 숫자형 벡터 생성
b = np.array(["apple", "banana", "orange"]) # 문자형 벡터 생성
c = np.array([True, False, True, True]) # 논리형 벡터 생성

a
b
c
type(a)
a[3]
a[2:]
a[1:4]

b = np.empty(3)
b
b[0]=1
b[1]=4
b[2]=10
b



vec1 = np.array([1,2,3,4,5])
vec2 = np.arange(100)
vec3 = np.arange(1,101)
vec4 = np.arange(1,101,0.5)
lin1 = np.linspace(0,10,5)
lin2 = np.linspace(0,1,5, endpoint=False)
lin1
lin2

np.repeat(3,5)
np.repeat(vec1,5)

vec5 = np.arange(-100,1,1)
vec5
vec6 = np.arange(0,-100,-1)
vec6


# repeat vs tile
vec1
np.repeat(vec1,3)
np.tile(vec1,3)

vec1 * 3 #원소별로 사칙연산 가능하다

max(vec1)
sum(vec1)

#  35672 이하 홀수들의 합
vec7 = np.arange(1,35673,2)
vec7
sum(vec7)
np.arange(1,35673,2).sum()

len(vec2)
vec2.shape

# 2차원 배열
b = np.array([[1,2,3],[4,5,6]])
b
b.shape
b.size

a = np.array([1,2])
b = np.array([1,2,3,4])
a + b #길이가 다르면 계산이 되지 않는다

b==3 #비교 연산자, 벡터안에 숫자 3과 같은 값이 있니?

# 35672보다 작은 수 중에서 7로 나눠서 나머지가 3인 정수의 갯수?
c = np.arange(1,35673)
c
c % 7
sum(c % 7 == 3) #True는 1로 계산 된다.


lec 5
