# 균일확률변수 만들기
import numpy as np

np.random.rand(2)

def X(num):
    return np.random.rand(num) 

X(3)


# 베르누이 확률변수 모수 : p 만들기
# 가질 수 있는 값은 2개

def Y(num, p):
    x = np.random.rand(num)
    return np.where(x < p,1,0)
Y(100, 0.5)

# 아래 2개는 같은 말
sum(Y(100, 0.5)) / 100
Y(100, 0.5).mean()

sum(Y(10000, 0.5)) / 10000
Y(10000000, 0.5).mean()



# 새로운 확률 변수
# 가질 수 있는 값 : 0,1,2
# 20%, 50%, 30%

p = np.array([0.2,0.5,0.3])

def Z(p):
    x = np.random.rand(1)
    p_cumsum = p.cumsum()
    return np.where(x < p_cumsum[0], 0, np.where(x < p_cumsum[1],1,2))

Z(p)

# 가질 수 있는 값 : 0,1,2,3
# 15%, 25%, 30%, 30%

def TH(p):
    x = np.random.rand(1)
    p_cumsum = p.cumsum()
    return np.where(x < p_cumsum[0],0,
           np.where(x < p_cumsum[1],1,
           np.where(x < p_cumsum[2],2,3)))
p = np.array([0.15,0.25,0.3,0.3])           

TH(p)

# E[X], 노션 참고
sum(np.arange(4) * (np.array([1,2,2,1]) / 6))





