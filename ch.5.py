import pandas as pd
import numpy as np

exam = pd.read_csv('data/exam.csv')
exam.head()
exam.shape

# 메서드 vs 속성(어트리뷰트)
# ()의 유무 차이

exam2 = exam.copy()
exam2 = exam2.rename(columns = {'nclass' :'class'})
exam2['total'] = exam2['math'] + exam2['english'] + exam2['science']
exam2.head()

exam2['test'] = np.where(exam2['total'] >= 200, 'Pass', 'Fail')
exam2.head()

import matplotlib.pyplot as plt
exam2['test'].value_counts().plot.bar(rot=0)
plt.show()
plt.clf()


exam2['test2'] = np.where(exam2['total'] >= 200, 'A',
                 np.where(exam2['total'] >= 100, 'B',
                 'C'))
                 
exam2.head()

exam2['test2'].isin(['A'])


exam = pd.read_csv('data/exam.csv')

# 조건에 맞는 행을 걸러내는 코드
exam.query('nclass == 1')
exam.query('nclass != 1')
exam.query('math > 50')
exam.query('math < 50')
exam.query('nclass == 1 & math >= 50')
exam.query('math >= 90 | english >= 90')
exam.query('nclass in [1,3,5]')
exam.query('nclass not in [1,3,5]')
# exam[~exam['nclass'].isin([1,2])]



exam[['math','english']]

# exam['math']와 exam[['math']]의 차이는 시리즈이냐? df이냐? 이다.

exam.drop(columns = 'math')

exam.query('nclass == 1') \
          [['math','english']] \
          .head()
          
exam.sort_values('math', ascending = False)
exam.sort_values(['nclass','english'], ascending = [True,False])

exam.assign(total = exam['math'] + exam['english'] + exam['science']) \
    .sort_values('total')
    
exam2 = pd.read_csv('data/exam.csv')

exam2 = exam2.assign(
    total = lambda x: x['math'] + x['english'] + x['science'])
exam2


# 그룹을 나눠 요약을 하는 .groupby().agg() 콤보
exam2.agg(mean_math = ('math','mean'))
exam2.groupby('nclass', as_index = False) \
     .agg(mean_math = ('math','mean'))


exam2.groupby('nclass') \
     .agg(mean_math = ('math','mean'),
          mean_english = ('english','mean'),
          mean_science = ('science','mean'))

mpg = pd.read_csv('data/mpg.csv')
mpg

mpg.query('category == "suv"') \
   .assign(total = (mpg['hwy'] + mpg['cty']) / 2) \
   .groupby('manufacturer') \
   .agg(mean_tot = ('total','mean')) \
   .sort_values('mean_tot', ascending = False) \
   .head()
