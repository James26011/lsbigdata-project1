# 숙제 확인 코드

# 구글 시트 불러오기
import pandas as pd

gsheet_url = "https://docs.google.com/spreadsheets/d/1RC8K0nzfpR3anLXpgtb8VDjEXtZ922N5N0LcSY5KMx8/gviz/tq?tqx=out:csv&sheet=Sheet2"
df = pd.read_csv(gsheet_url)
df.head()

# 랜덤하게 2명을 뽑아서 보여주는 코드
import numpy as np

np.random.seed(20240730)
np.random.choice(df["이름"], 2, replace=False)

===================================================================================

# !pip install pyreadstat
import pandas as pd
import numpy as np
import seaborn as sns

raw_welfare=pd.read_spss("./data/koweps/Koweps_hpwc14_2019_beta2.sav")
raw_welfare

welfare=raw_welfare.copy()
welfare.shape

welfare=welfare.rename(
    columns = {
        "h14_g3": "sex",
        "h14_g4": "birth",
        "h14_g10": "marriage_type",
        "h14_g11": "religion",
        "p1402_8aq1": "income",
        "h14_eco9": "code_job",
        "h14_reg7": "code_region"
    }
)

welfare=welfare[["sex", "birth", "marriage_type",
                "religion", "income", "code_job", "code_region"]]
welfare.shape

welfare["sex"].dtypes
welfare["sex"].value_counts()
# welfare["sex"].isna().sum()

welfare["sex"] = np.where(welfare["sex"] == 1,'male', 'female')
welfare["sex"].value_counts()


welfare["income"].isna().sum()

sex_income=welfare.dropna(subset="income") \
                  .groupby("sex", as_index=False) \
                  .agg(mean_income = ("income", "mean"))

sex_income

import seaborn as sns

sns.barplot(data=sex_income, x="sex", y="mean_income",
            hue="sex")
plt.show()
plt.clf()

# 숙제: 위 그래프에서 각 성별 95% 신뢰구간 계산후 그리기
# 위 아래 검정색 막대기로 표시

welfare["birth"].describe()
sns.histplot(data=welfare, x="birth")
plt.show()
plt.clf()

welfare["birth"].isna().sum()

welfare=welfare.assign(age = 2019 - welfare["birth"] + 1)
welfare["age"]
sns.histplot(data=welfare, x="age")
plt.show()
plt.clf()

age_income=welfare.dropna(subset="income") \
                    .groupby("age", as_index=False) \
                    .agg(mean_income = ("income", "mean"))

sns.lineplot(data=age_income, x="age", y="mean_income")
plt.show()
plt.clf()

# 나이별 income 칼럼 na 개수 세기!
welfare["income"].isna().sum()

welfare["income"].isna()
my_df=welfare.assign(income_na=welfare["income"].isna()) \
                        .groupby("age", as_index=False) \
                        .agg(n = ("income_na", "sum"))


sns.barplot(data = my_df, x="age", y="n")
plt.show()
plt.clf()

# 나이대별 수입 분석
# cut
bin_cut=np.array([0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119])
welfare=welfare.assign(age_group = pd.cut(welfare["age"], 
                bins=bin_cut, 
                labels=(np.arange(12) * 10).astype(str) + "대"))

# np.version.version
# (np.arange(12) * 10).astype(str) + "대"

age_income=welfare.dropna(subset="income") \
                    .groupby("age_group", as_index=False) \
                    .agg(mean_income = ("income", "mean"))

age_income
sns.barplot(data=age_income, x="age_group", y="mean_income")
plt.show()
plt.clf()

# 판다스 데이터 프레임을 다룰 때, 변수의 타입이
# 카테고리로 설정되어 있는 경우, groupby+agg 콤보
# 안먹힘. 그래서 object 타입으로 바꿔 준 후 수행
welfare["age_group"]=welfare["age_group"].astype("object")

def my_f(vec):
    return vec.sum()

sex_age_income = \
    welfare.dropna(subset="income") \
    .groupby(["age_group", "sex"], as_index=False) \
    .agg(top4per_income=("income", lambda x: my_f(x)))
    
sex_age_income


sex_age_income = \
    welfare.dropna(subset="income") \
    .groupby(["age_group", "sex"], as_index=False) \
    .agg(top4per_income=("income", lambda x: np.quantile(x, q=0.96)))
    



sex_age_income

sns.barplot(data=sex_age_income,
            x="age_group", y="mean_income", 
            hue="sex")
plt.show()
plt.clf()

# ============9-6장============
welfare['code_job']
list_job = pd.read_excel('./data/koweps/Koweps_Codebook_2019.xlsx', sheet_name='직종코드')
list_job.head()

welfare = welfare.merge(list_job, how='left', on='code_job')
welfare.dropna(subset=['job','income'])[['income','job']]

job_income = welfare.dropna(subset = ['job','income']) \
                    .query('sex == "male"') \
                    .groupby('job',as_index = False) \
                    .agg(mean_income = ('income','mean')) \
                    .sort_values('mean_income', ascending = False) \
                    .head(10)
                    
job_income

top10 = job_income.sort_values('mean_income',ascending = False).head(10)
top10
                    
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family' :'Malgun Gothic'})
sns.barplot(data = top10, y = 'job', x = 'mean_income', hue = 'job')
plt.show()
plt.clf()

bottom10 = job_income.sort_values('mean_income').head(10)
bottom10

sns.barplot(data = bottom10, y = 'job', x = 'mean_income', hue = 'job') \
   .set(xlim = [0,800])
plt.show()
plt.clf()

# 9-8
welfare['marriage_type']
df = welfare.query('marriage_type != 5') \
            .groupby('religion',as_index=False) \
            ['marriage_type'] \
            .value_counts(normalize = True) \ # 핵심!

df 
            
rel_df = df.query('marriage_type ==1') \
           .assign(proportion = df['proportion']*100) \
           .round(1)


rel_df






