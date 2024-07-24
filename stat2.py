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


