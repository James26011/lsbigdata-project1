import pandas as pd

test1 = pd.DataFrame({'id' : [1,2,3,4,5],
                      'midterm' : [60,80,70,90,85]})
                      
test2 = pd.DataFrame({'id' : [1,2,3,4,5],
                      'final' : [70,83,65,95,85]})
test1
test2


total = pd.merge(test1, test2, how = 'left', on = 'id')
total

test1 = pd.DataFrame({'id' : [1,2,3,4,5],
                      'midterm' : [60,80,70,90,85]})
                      
test2 = pd.DataFrame({'id' : [1,2,3,40,5],
                      'final' : [70,83,65,95,85]})
                      
# 40번 학생이 없지만, 기준점이 left라서 id 기준은 1,2,3,4,5 이다
# 그래서 4번 학생의 기말이 Nan으로 표시 된다
total = pd.merge(test1, test2, how = 'left', on = 'id')
total
# 반대로 기준을 오른쪽으로 하면 40번 학생이 나타나고 중간 점수가 Nan이 된다.

total = pd.merge(test1, test2, how = 'inner', on = 'id')
total
# inner은 id의 교집합을 남기고 출력한다

total = pd.merge(test1, test2, how = 'outer', on = 'id')
total
# outer은 id의 합집합을 출력한다.


name = pd.DataFrame({'nclass' :[1,2,3,4,5],
                     'teacher' : ['kim','lee','park','choi','jung']})
name

exam = pd.read_csv('data/exam.csv')
exam_new = pd.merge(exam, name, how = 'left', on = 'nclass')
exam_new

# 세로로 쌓기
score1 = pd.DataFrame({'id' : [1,2,3,4,5],
                      'score' : [60,80,70,90,85]})
                      
score2 = pd.DataFrame({'id' : [6,7,8,9,10],
                      'score' : [70,83,65,95,85]})
score1
score2

score_all = pd.concat([score1, score2])
score_all

pd.concat([score1, score2], axis =1)





