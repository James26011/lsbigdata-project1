x = 15
print(x, "는 ", type(x), " 형식입니다.", sep='')

# 문자형 데이터 예제
a = "Hello, world!"
b = 'python programming'

# 여러 줄 문자열
ml_str = """This is
a multi-line
string"""

print(a, type(a))
print(b, type(b))
print(ml_str, type(ml_str))

# 문자열 결합
greeting = "안녕" + " " + "파이썬!"
print("결합 된 문자열:", greeting)

# 문자열 반복
laugh = "하" * 3
print("반복 문자열:", laugh)

# 리스트
fruits = ['apple', 'banana', 'cherry']
type(fruits)

numbers = [1, 2, 3, 4, 5]
MixedList: [1, 'Hello', [1, 2, 3]]

# 튜플
b1 = (42)
type(b1)
b2 = (42,)
type(b2)

# 인덱스
a_tp = (10,20,30)
a_tp[0]
a_tp[1] = 25

a_list = [10,20,30,40,50]
a_list[1:3]

# 사용자 정의 함수
def min_max(numbers):
  return min(numbers), max(numbers)

a = [1, 2, 3, 4, 5]
result = min_max(a)
result[0] = 4
type(result) # 튜플로 반환

# 딕셔너리 생성 예제
person = {
  'name': 'James',
  'age': (29,21),
  'city': ['Busan','Seoul'] #리스트 가능, 튜플 가능
}
print("Person:", person2)

person.get('name')
age = person.get('age')
person.get('age')[1] #2번째 나이 출력
age[1]


# 집합 생성 예제
fruits = {'apple', 'banana', 'cherry', 'apple'}
print("Fruits set:", fruits) # 중복 'apple'은 제거됨
type(fruits)

# 빈 집합 생성
empty_set = set()
print("Empty set:", empty_set)

empty_set.add('apple')
empty_set.add('banana')
empty_set.add('apple')
empty_set.remove('banana')
empty_set.discard('cherry')
empty_set.remove('cherry')

# 논리형 데이터 예제
p = True
q = False
print(p, type(p))
print(q, type(q))
print(p + p) # True는 1로, False는 0으로 계산됩니다.

age = 10
age > 5

a=3
if (a == 2):
  print("a는 2와 같습니다.")
else:
  print("a는 2와 같지 않습니다.")
  
# 숫자형을 문자열형으로 변환
num = 123
str_num = str(num)
print("문자열:", str_num, type(str_num))

# 문자열형을 숫자형(실수)으로 변환
num_again = float(str_num)
print("숫자형:", num_again, type(num_again))
  
set_example = {'a', 'b', 'c'}
# 딕셔너리로 변환 시, 일반적으로 집합 요소를 키 또는 값으로 사용
dict_from_set = {key: True for key in set_example}
print("Dictionary from set:", dict_from_set)
