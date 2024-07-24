import pandas as pd
import numpy as np

df = pd.DataFrame({'sex' : ['M','F',np.nan,'M','F'],
                   'score' : [5,4,3,4,np.nan]})
df
df['score'] + 1

pd.isna(df).sum()

df.dropna(subset = 'score')
df.dropna(subset = ['score','sex'])

df.loc[df['score'] == 3.0, ['score']] = 4
df

exam = pd.read_csv('data/exam.csv')

# exam.loc[[행 인덱스],[열 인덱스]]
exam.loc[[2,7,14],['math']] = np.nan

# iloc로 하면 숫자로 인덱싱 가능
exam.iloc[[2,7,14], 2] = np.nan
exam
exam.isna().sum()

# 수학점수 50점 이하인 학생들 점수 50점으로 상향 조정
exam.loc[exam['math'] <= 50, 'math']
exam.loc[exam['math'] <= 50, 'math'] = 50
exam

# 영어점수 90점 이상을 90점으로 하향 조정 iloc 사용
exam.loc[exam['english'] >= 90, 'english']
exam.iloc[exam[exam['english'] >= 90].index, 3] = 90
exam.iloc[np.array(exam['english'] >= 90), 3] = 90
exam.iloc[np.where(exam['english'] >= 90)[0], 3] = 90
exam

# 수학 점수 50점 이하 "-" 로 변경
exam.loc[exam['math'] <= 50, 'math'] = '-'
exam

# -인 부분을 수학 평균 점수로 바꾸기
math_mean = exam.loc[(exam['math'] != '-'), 'math'].mean()
exam.loc[exam['math'] == '-','math'] = math_mean

math_mean = exam.query('math not in ["-"]')['math'].mean()
exam.loc[exam['math'] == '-','math'] = math_mean

math_mean = exam[exam['math'] != '-']['math'].mean()
exam.loc[exam['math'] == '-','math'] = math_mean

math_mean = exam[exam['math'] != '-']['math'].mean()
exam['math'] = exam['math'].replace('-', math_mean)
exam



