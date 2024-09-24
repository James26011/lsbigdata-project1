import numpy as np

## q1 

a = np.arange(2, 13)
b = np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]) / 36

mean = sum(a*b)

var = sum((a - mean) ** 2)/ 11 

## q2

x_q = mean * 2 + 3

var_q = var * 4


from scipy.stats import binom

import matplotlib.pyplot as plt
from scipy.stats import chi2
k = np.arange(0,30.1,0.1)
y = chi2.pdf(k, 6)
plt.plot(k, y)
plt.show()




mat_a = np.array([14, 4, 0, 10]).reshape(2,2)
mat_a

from scipy.stats import chi2_contingency

chi2, p, df, expected = chi2_contingency(mat_a)
chi2.round(3) # 검정통계량
p.round(4) # p-value

# 유의수준이 0.05이라면 p값이 작으므로 귀무가설 기각 / 그래서 두 변수는 독립이 아니다.

