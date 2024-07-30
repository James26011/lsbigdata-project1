import numpy as np
import pandas as pd

df = pd.DataFrame({'name' : ['김지훈','이유진','박동현','김민지'],
                   'english':[90,80,60,70],
                   'math': [50,60,100,20]})
df
type(df)

df['name']
sum(df['english'])/4

df = pd.DataFrame({'제품': ['사과','딸기','수박'],
                   '가격': [1800,1500,3000],
                   '판매량': [24,38,13]})
 df
 
sum(df['가격'])/3
sum(df['판매량'])/3

!pip install openpyxl
df_exam = pd.read_excel("data/excel_exam.xlsx")
df_exam

sum(df_exam['math']) /20
sum(df_exam['english'])/20
sum(df_exam['science'])/20

len(df_exam)
df_exam.shape
df_exam.size

df_exam['total'] = df_exam['math'] + df_exam['english'] + df_exam['science']
df_exam['mean'] = df_exam['total'] / 3
df_exam['mean']
df_exam

df_exam[df_exam['math']>50]

df_exam[(df_exam['math'] > 50) & (df_exam['english'] > 50)]


# 수학 평균 보다 높은 사람이 영어 평균이 낮은 사람은?
df_exam[(df_exam['math'] > sum(df_exam['math']/20)) & 
        (df_exam['english'] < sum(df_exam['english']/20))]
# 3반 학생들만 추출
df_exam[df_exam['nclass'] == 3]
               [['math','english','science']]


a = np.array([4,2,5,3,6])
a[2]

df_exam[:10]

df_exam[7:16:2]
df_exam.sort_values(['nclass','math'],ascending = [True, False])

np.where(a > 3,'Up','Down') #진실이면 up, 거짓이면 down

df_exam['updown'] = np.where(df_exam['math'] > 50,'Up','Down')
df_exam
