import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

# 데이터 불러오기
house_train=pd.read_csv("./house/train.csv")
house_test=pd.read_csv("./house/test.csv")

################3 결측값 채우기
## train
## 숫자형 채우기
quantitative = house_train.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    house_train[col].fillna(house_train[col].mean(), inplace=True)
house_train[quant_selected].isna().sum()

## 범주형 채우기
qualitative = house_train.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    house_train[col].fillna("unknown", inplace=True)
house_train[qual_selected].isna().sum()

## test
## 숫자형 채우기
quantitative = house_test.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected2 = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected2:
    house_test[col].fillna(house_test[col].mean(), inplace=True)
house_train[quant_selected].isna().sum()

## 범주형 채우기
qualitative = house_train.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    house_test[col].fillna("unknown", inplace=True)
house_test[qual_selected].isna().sum()

train_n=len(house_train)

cols = house_train.select_dtypes(include='object').columns.tolist()



############## 데이터 합치기
df=pd.concat([house_train, house_test]).reset_index(drop=True)
df

#더미 작업
d=pd.get_dummies(df,columns=cols)

train_d = d.iloc[:1460,:]
test_d = d.iloc[1460:,:]
test_d = test_d.drop(columns='SalePrice')

train_dx = train_d.drop(columns='SalePrice')
train_dy = train_d['SalePrice']





############## 알파값 찾아보기
kf = KFold(n_splits=5, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, train_dx, train_dy, cv = kf,
                                     n_jobs=-1, scoring = "neg_mean_squared_error").mean())
    return(score)

# 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(0, 200, 0.01)
mean_scores = np.zeros(len(alpha_values))

k=0
for alpha in alpha_values:
    lasso = Lasso(alpha=alpha)
    mean_scores[k] = rmse(lasso)
    k += 1

# 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)





############### 학습해서 제출하기
model= Lasso(alpha=169.84)
model.fit(train_dx, train_dy)
y_pred = model.predict(test_d)

submit = pd.read_csv('data/sample_submission.csv')
submit['SalePrice'] = y_pred