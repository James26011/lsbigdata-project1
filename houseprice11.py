# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

## 필요한 데이터 불러오기
house_train=pd.read_csv("./house/train.csv")
house_test=pd.read_csv("./house/test.csv")
sub_df=pd.read_csv("./house/sample_submission.csv")

## NaN 채우기
# 각 숫치형 변수는 평균 채우기
# 각 범주형 변수는 Unknown 채우기
house_train.isna().sum()
house_test.isna().sum()

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


# test 데이터 채우기
## 숫자형 채우기
quantitative = house_test.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    house_test[col].fillna(house_train[col].mean(), inplace=True)
house_test[quant_selected].isna().sum()

## 범주형 채우기
qualitative = house_test.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    house_test[col].fillna("unknown", inplace=True)
house_test[qual_selected].isna().sum()


house_train.shape
house_test.shape
train_n=len(house_train)

# 통합 df 만들기 + 더미코딩
# house_test.select_dtypes(include=[int, float])

df = pd.concat([house_train, house_test], ignore_index=True)
# df.info()
df = pd.get_dummies(
    df,
    columns= df.select_dtypes(include=[object]).columns,
    drop_first=True
    )
df

# train / test 데이터셋
train_df=df.iloc[:train_n,]
test_df=df.iloc[train_n:,]

## 이상치 탐색
train_df=train_df.query("GrLivArea <= 4500")

## train
train_x=train_df.drop("SalePrice", axis=1)
train_y=train_df["SalePrice"]

## test
test_x=test_df.drop("SalePrice", axis=1)

from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

eln_model= ElasticNet()
rf_model= RandomForestRegressor(n_estimators=100)

# 그리드 서치 for ElasticNet
param_grid={
    'alpha': [0.1, 1.0, 10.0, 100.0],
    'l1_ratio': [0, 0.1, 0.5, 1.0]
}
grid_search=GridSearchCV(
    estimator=eln_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5
)
grid_search.fit(train_x, train_y)
best_eln_model=grid_search.best_estimator_

# 그리드 서치 for RandomForests
param_grid={
    'max_depth': [3, 5, 7],
    'min_samples_split': [20, 10, 5],
    'min_samples_leaf': [5, 10, 20, 30],
    'max_features': ['sqrt', 'log2', None]
}
grid_search=GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5
)
grid_search.fit(train_x, train_y)
grid_search.best_params_
best_rf_model=grid_search.best_estimator_


# 스택킹
y1_hat=best_eln_model.predict(train_x) # test 셋에 대한 집값
y2_hat=best_rf_model.predict(train_x) # test 셋에 대한 집값

train_x_stack=pd.DataFrame({
    'y1':y1_hat,
    'y2':y2_hat
})

from sklearn.linear_model import Ridge

rg_model = Ridge()
param_grid={
    'alpha': np.arange(0, 10, 0.01)
}
grid_search=GridSearchCV(
    estimator=rg_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5
)
grid_search.fit(train_x_stack, train_y)
grid_search.best_params_
blander_model=grid_search.best_estimator_

blander_model.coef_
blander_model.intercept_

pred_y_eln=best_eln_model.predict(test_x) # test 셋에 대한 집값
pred_y_rf=best_rf_model.predict(test_x) # test 셋에 대한 집값

test_x_stack=pd.DataFrame({
    'y1': pred_y_eln,
    'y2': pred_y_rf
})

pred_y=blander_model.predict(test_x_stack)


# ============================
from xgboost import XGBRegressor

# XGBoost 모델 추가
xgb_model = XGBRegressor()

# 그리드 서치 for XGBoost
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5
)

grid_search.fit(train_x, train_y)
best_xgb_model = grid_search.best_estimator_

# XGBoost 모델을 사용한 예측
y3_hat = best_xgb_model.predict(train_x)

# 스태킹 데이터셋에 XGBoost 결과 추가
train_x_stack['y3'] = y3_hat

# 리지 회귀 모델에 새로운 변수(y3) 추가 후 재학습
rg_model = Ridge()
param_grid = {
    'alpha': np.arange(0, 10, 0.01)
}

grid_search = GridSearchCV(
    estimator=rg_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5
)

grid_search.fit(train_x_stack, train_y)
grid_search.best_params_
blender_model = grid_search.best_estimator_

# 테스트 셋에 대해 예측
pred_y_eln = best_eln_model.predict(test_x) # ElasticNet 예측
pred_y_rf = best_rf_model.predict(test_x) # RandomForest 예측
pred_y_xgb = best_xgb_model.predict(test_x) # XGBoost 예측

test_x_stack = pd.DataFrame({
    'y1': pred_y_eln,
    'y2': pred_y_rf,
    'y3': pred_y_xgb
})

# 최종 예측
final_pred_y = blender_model.predict(test_x_stack)


# =============================






# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y
sub_df

# # csv 파일로 내보내기
sub_df.to_csv("./house/sub10.csv", index=False)