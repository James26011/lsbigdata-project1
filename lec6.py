import numpy as np

# 두 개의 벡터를 합쳐 행렬 생성
matrix = np.column_stack((np.arange(1, 5),
np.arange(12, 16)))

matrix = np.vstack((np.arange(1, 5),
np.arange(12, 16)))

print("행렬:\n", matrix)
type(matrix)


np.zeros(5)
np.zeros([5,4])

np.arange(1,5).reshape([2,2])
np.arange(1,7).reshape((2,3))
np.arange(1,7).reshape((2,-1))

# Q. 0에서 99까지 수 중 랜덤하게 50개 숫자를 뽑아서 5-10 행렬을 만드시오.
a = np.random.randint(1,100,50)
a.reshape(5,10)

# order 행, 열 옵션
np.arange(1,21).reshape((4,5), order = 'F') #세로 방향
np.arange(1,21).reshape(4,5) # 가로 방향

mat_a = np.arange(1,21).reshape((4,5), order = 'F')

# 인덱싱
mat_a[0,0]
mat_a[1,1]
mat_a[0:2,3]
mat_a[1:3,1:4]

mat_a[3,] # 가능은 함
mat_a[3,:] # 이게 더 파이썬 스러움
mat_a[3,::2]


mat_b = np.arange(1,101).reshape(20,-1)
mat_b[1::2,:]
mat_b[[1,4,6,14],:]

x = np.arange(1,11).reshape((5,2))*2
x[[True,True,False,False,True],0]

mat_b[:,1]
mat_b[:,1:2]

# 필터링
mat_b[mat_b[:,1] % 7 == 0,:] # bool 연산자로 필터링 가능


# 사진은 행렬이다.
import numpy as np
import matplotlib.pyplot as plt

# 난수 생성하여 3x3 크기의 행렬 생성
np.random.seed(2024)
img1 = np.random.rand(3, 3)
print("이미지 행렬 img1:\n", img1)

plt.imshow(img1, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()


a = np.random.randint(1,10,20).reshape(4,-1)
a / 9

import urllib.request
img_url = "https://bit.ly/3ErnM2Q"
urllib.request.urlretrieve(img_url, "jelly.png")

!pip install imageio 
import imageio
import numpy as np

# 이미지 읽기
jelly = imageio.imread("jelly.png")
print("이미지 클래스:", type(jelly))
print("이미지 차원:", jelly.shape)
print("이미지 첫 4x4 픽셀, 첫 번째 채널:\n", jelly[:4, :4, 0])
plt.show

# 젤리 사진은 4장의 사진을 겹쳐놓은 것
jelly[:,:,0]
jelly[:,:,1]
jelly[:,:,2]
jelly[:,:,3]

plt.imshow(jelly)
plt.axis('off')
plt.show()

plt.imshow(jelly)

plt.imshow(jelly[:,:,0].transpose())
plt.imshow(jelly[:,:,0]) # R
plt.imshow(jelly[:,:,1]) # G
plt.imshow(jelly[:,:,2]) # B
plt.imshow(jelly[:,:,3]) # 투명도
plt.axis('off') # 축 정보 없애기
plt.show()
plt.clf()


# 두 개의 2x3 행렬 생성
mat1 = np.arange(1, 7).reshape(2, 3)
mat2 = np.arange(7, 13).reshape(2, 3)

# 3차원 배열로 합치기
my_array = np.array([mat1, mat2])
print("3차원 배열 my_array:\n", my_array)

my_array[:,:,[0,2]]
my_array[:,0,:]
my_array[0,1,1:3]

my_array.shape # 2장, 2행 3열

my_array2 = np.array([my_array, my_array])
my_array2
my_array2.shape #2행 3열인데 2장 겹쳐진게 2번 겹쳣다

# 첫 번째 2차원 배열 선택
first_slice = my_array[0, :, :]
print("첫 번째 2차원 배열:\n", first_slice)
# 두 번째 차원의 세 번째 요소를 제외한 배열 선택
filtered_array = my_array[:, :, :-1]
print("세 번째 요소를 제외한 배열:\n", filtered_array)

mat_x = np.arange(1,101).reshape((5,5,4)) # 5장 5행 4열
mat_y = np.arange(1,101).reshape((-1,5,2)) # 자동, 5행 2열



# 넘파이 배열 메서드
a = np.array([[1,2,3],[4,5,6]])

a.sum()
a.sum(axis = 0)
a.sum(axis = 1)

a.mean()
a.mean(axis = 0)
a.mean(axis = 1)

mat_b = np.random.randint(0,100,50).reshape((5,-1))
mat_b
mat_b.max()

# 행별 가장 큰 수?
mat_b.max(axis=1)

# 열별 가장 큰 수?
mat_b.max(axis=0)

a = np.array([1,3,2,5])
a.cumsum()
mat_b.cumsum(axis = 1)

mat_b.cumprod(axis=1)

mat_b.reshape((2,5,5))
mat_b.flatten()

d = np.array([1, 2, 3, 4, 5])
print("클립된 배열:", d.clip(2, 4))

print("리스트:", d.tolist())



