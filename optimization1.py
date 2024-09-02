# y=(x-2)^2 + 1 그래프 그리기
# 점을 직선으로 이어서 표현
import matplotlib.pyplot as plt
import numpy as np

k=6
x = np.linspace(-4, 8, 100)
y = (x - 2)**2 + 1
# plt.scatter(x, y, s=3)
plt.plot(x, y, color="black")
plt.xlim(-4, 8)
plt.ylim(0, 15)

# f'(x)=2x-4
# k=4의 기울기
l_slope=2*k - 4
f_k=(k-2)**2 + 1
l_intercept=f_k - l_slope * k

# y=slope*x+intercept 그래프
line_y=l_slope*x + l_intercept
plt.plot(x, line_y, color="red")

x = 10
y = x - (0.9 * 2*x)

k = y - (0.9 * 2*y)

# y = x^2 경사하강법
# 초기값:10, 델타: 0.9
x = 10
lstep = np.arange(100,0,-1)*0.01
for i in range(100):
    x-=lstep[i]*(2*x) # x = x-lstep*(2*x)
print(x)


