# 리스트 예제
fruits = ["apple", "banana", "cherry"]
numbers = [1, 2, 3, 4, 5]
mixed = [1, "apple", 3.5, True]
print("과일 리스트:", fruits)
print("숫자 리스트:", numbers)
print("혼합 리스트:", mixed)

# 빈 리스트 생성
empty_list1 = []
empty_list2 = list()
print("빈 리스트 1:", empty_list1)
print("빈 리스트 2:", empty_list2)

# 초기값을 가진 리스트 생성
numbers = [1, 2, 3, 4, 5]
range_list = list(range(5))
print("숫자 리스트:", numbers)
print("range() 함수로 생성한 리스트:", range_list)


# 리스트 내포(comprehension)
# 1. 대괄호로 쌓여져있다. = 리스트다.
# 2. 넣고 싶은 수식표현을 x를 사용해서 표현
# 3. for .. in .. 을 사용해서 원소정보제공

list(range(10))
squares = [x**2 for x in range(10)]
squares

# 리스트 가능
l1 = [3,5,2,15]
my_squares = [x**3 for x in l1]
my_squares

# 넘파이 배열 가능
import numpy as np
a = np.array([3,5,2,15])
my_squares2 = [x**3 for x in a]
my_squares2

# 판다스 시리즈 가능
import pandas as pd
exam = pd.read_csv('data/exam.csv')
my_squares3 = [x**3 for x in exam['math']]
my_squares3

# 리스트 합치기
3 + 2
'안녕' + '하세요'
# 리스트 연결
list1 = [1, 2, 3]
list2 = [4, 5, 6]
combined_list = list1 + list2
print("연결된 리스트:", combined_list)

(list1 * 3) + (list2 * 2)


numbers = [5,2,3]
repeated_list = [x for x in numbers for _ in range(3)]
repeated_list = [x for x in numbers for _ in [4,2,1]]
repeated_list
# _의 의미 = 앞에 나온 값을 가르킴
5+4
_ + 6 # _는 9룰 의미

# 값 생략, 자리 차지 placeholder
a, _, b = (1,2,4)
a;b
_
del _

# for 루프 문법
# for i in 범위:
#   작동방식
for x in [4,1,2,3]:
    print(x)
x

for i in range(5):
    print(i**2)
i

#리스트를 하나 만들어서, for 루프를 사용해서
# 2,4,6,8,...,20의 수를 채워 넣기
range(1,11)

i for i in range(2,21,2):
    print(y)

# 1.
mylist = []
for i in range(1, 11):
    mylist.append(i*2)
mylist

# 2
mylist = [0] * 10
for i in range(10):
    mylist[i] = 2 * (i + 1) 
mylist

# 퀴즈: mylist_b의 홀수번째 위치에 있는 숫자들만 mylist에 가져오기
mylist_b = [2,4,6,80,10,12,24,35,23,20]
mylist = [0]*5

for i in range(5):
    mylist[i] = mylist_b[2*i]
mylist

#리스트 컴프리헨션으로 바꾸는 방법
#바깥은 무조건 대괄호로 묶어줌 : 리스트 반환하기 위해서.
#for 루프의 : 는 생략한다.
#실행부분을 먼저 써준다.
mylist = []
mylist.append

[i*2 for i in range(1,11])
[x for x in numbers]

for i in range(5):
    print('hello')

for i in [0,1,2]:
    for j in [4,5,6]:
        print(i,j)

numbers = [5,2,3]
for i in numbers:
    for j in range(4):
        print(i,j)

numbers = [5,2,3]
for i in numbers:
    for j in range(4):
        print(i)

# [print(i) for i in numbers for j in range(4)]
[i for i in numbers for j in range(4)]



# 원소 체크
fruits = ["apple", "banana", "cherry"]
fruits
'banana' in fruits

# [x == "banana" for x in fruits]
mylist = []
for x in fruits:
    mylist.append(x == 'banana')
mylist


# 바나나의 위치를 뱉어내게 하려면?
fruits = ["apple", "apple", "banana", "cherry"]
import numpy as np

fruits = np.array(fruits)
int(np.where(fruits == 'banana')[0][0])

# 원소 거꾸로 써주는 reverse()
fruits = ["apple", "apple", "banana", "cherry"]
fruits.reverse()
fruits

# 원소 맨 끝에 붙여주기
fruits.append('pineapple')
fruits

# 원소 삽입
fruits.insert(2,'test')
fruits

# 원소 제거
fruits.remove('test')
fruits # 중복되는 값이 있으면 인덱스가 빠른 것 부터 지운다.


import numpy as np
# 넘파이 배열 생성
fruits = np.array(["apple", "banana", "cherry", "apple", "pineapple"])

# 제거할 항목 리스트
items_to_remove = np.array(["banana", "apple"])

# 불리언 마스크 생성
mask = ~np.isin(fruits, items_to_remove)

# 불리언 마스크를 사용하여 항목 제거
filtered_fruits = fruits[mask]
print("remove() 후 배열:", filtered_fruits)

