import pandas as pd
import numpy as np

tab3 = pd.read_csv('./data/tab3.csv')    
tab3

tab1 = pd.DataFrame({'id' : np.arange(1,13),
                     'score' : tab3['score']})
tab1

tab2 = tab1.assign(gender =['female']*7 + ['male']*5)
tab2


# 1 표본 t 검정 (그룹 1개)
# 귀무가설 vs 대립가설
# H0: mu = 10 vs Ha: mu != 10
# 유의수준 5%로 설정
from scipy.stats import ttest_1samp

result = ttest_1samp(tab1['score'], popmean=10, alternative='two-sided')
t_value = result[0] # t검정 통계량
p_value = result[1] # 유의확률 (p-value)
tab1['score'].mean()
result.pvalue
result.statistic
result.df
# 귀무가설이 참일 때, 11.53이 관찰될 확률이 6.48%이므로, 이것은 우리가 생각하는 보기 힘들다고 판단하는 기준인
# 0.05(유의수준) 보다 크므로, 귀무가설을 거짓이라 판단하기 힘들다.
# 유의학률 0.0648이 유의수준 0.05보다 크므로 귀무가설을 기각하지 못한다.
# 95% 신뢰구간 구하기
ci= result.confidence_interval(confidence_level=0.95)
ci[0]
ci[1]


# 2 표본 t 검정 (그룹 2개)
# 분산이 같은 경우 : 독립 2표본 t검정
# 분산이 다를 경우 :웰치스 t 검정
# 귀무가설 vs 대립가설
# H0: mu_m = mu_f vs Ha: mu_m > mu_f
# 유의수준 1%로 설정, 두 그룹 분산은 같다고 가정
from scipy.stats import ttest_ind

f_tab2 = tab2[tab2['gender'] == 'female']
m_tab2 = tab2[tab2['gender'] == 'male']

# alternative = 'less' 의 의미는 대립가설이 첫번째 입력그룹의 평균이 두번쨰 입력 그룹 평균보다 작다고
# 설정된 경우를 나타낸다.
result = ttest_ind(f_tab2['score'], m_tab2['score'], 
                   alternative = 'less', equal_var=True)

tab2['score'].mean()
result.statistic # t검정 통계량
result.pvalue # 유의확률 (p-value)
result.df


# 대응표본 t 검정 (짝지을 수 있는 표본)
# 귀무가설 vs 대립가설
# H0: mu_before = mu_after vs Ha: mu_after > mu_before
# H0: mu_d = 0 vs Ha: mu_d > 0
# mu_d = mu_after - mu_before
# 유의수준 1%로 설정
tab3
tab3_data = tab3.pivot_table(index='id',columns='group',values='score')
tab3_data['score_diff'] = tab3_data['after'] - tab3_data['before']
test3_data = tab3_data[['score_diff']]
test3_data

result = ttest_1samp(test3_data['score_diff'], popmean=0, alternative='greater')
t_value = result[0] # t검정 통계량
p_value = result[1] # 유의확률 (p-value)

result.pvalue
result.statistic
result.df

# pivot_table
tab3
tab3_data = tab3.pivot_table(index='id',columns='group',values='score')
tab3_data
# 다시 long-form 으로 변환
long_form = tab3_data.melt(
                           id_vars = 'id',
                           value_vars = ['before','after'],
                           var_name = 'group',
                           value_name = 'score')


                           
df = pd.DataFrame({'id':[1,2,3],
                   'A':[10,20,30],
                   'B':[40,50,60]})
df_long = df.melt(
        id_vars = 'id',
        value_vars = ['A','B'],
        var_name = 'group',
        value_name = 'score')
df_long

df_long.pivot_table(index = 'id',columns='group', values = 'score')
df_long.pivot_table(columns='group', values = 'score')

# 연습
import seaborn as sns
tips = sns.load_dataset('tips')
tips = tips.reset_index(drop = False)


tips_data = tips.pivot_table(index = 'index',columns = 'day', values='tip')
tips_data





