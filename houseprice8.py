# 원하는 변수를 사용해서 회귀모델을 만들고, 제출할것!
# 원하는 변수 2개
# 회귀모델을 통한 집값 예측

# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

## 필요한 데이터 불러오기
house_train=pd.read_csv("./house/train.csv")
house_test=pd.read_csv("./house/test.csv")
sub_df=pd.read_csv("./house/sample_submission.csv")

## 이상치 탐색 (여기 넣으면 안됨!)
# house_train=house_train.query("GrLivArea <= 4500")

house_train.shape
house_test.shape
train_n=len(house_train)

# 통합 df 만들기 + 더미코딩
df = pd.concat([house_train, house_test], ignore_index=True)
df = pd.get_dummies(
    df,
    columns= ["Neighborhood"],
    drop_first=True
    )
df

# train / test 데이터셋
train_df=df.iloc[:train_n,]
test_df=df.iloc[train_n:,]

# Validation 셋(모의고사 셋) 만들기
np.random.seed(42)
val_index=np.random.choice(np.arange(train_n), size=438, replace=False)
val_index

# train => valid / train 데이터셋
valid_df=train_df.loc[val_index]  # 30%
train_df=train_df.drop(val_index) # 70%

## 이상치 탐색
train_df=train_df.query("GrLivArea <= 4500")

# x, y 나누기
# regex (Regular Expression, 정규방정식)
# ^ 시작을 의미, $ 끝남을 의미, | or를 의미
selected_columns=train_df.filter(regex='^GrLivArea$|^GarageArea$|^Neighborhood_').columns

## train
train_x=train_df[selected_columns]
train_y=train_df["SalePrice"]

## valid
valid_x=valid_df[selected_columns]
valid_y=valid_df["SalePrice"]

## test
test_x=test_df[selected_columns]

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(train_x, train_y)  # 자동으로 기울기, 절편 값을 구해줌

# 성능 측정 ()
y_hat=model.predict(valid_x)
np.sqrt(np.mean((valid_y-y_hat)**2))

## test 셋 결측치 채우기
test_x["GrLivArea"].isna().sum()
test_x["GarageArea"].isna().sum()
test_x=test_x.fillna(house_test["GarageArea"].mean())

pred_y=model.predict(test_x) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("./data/houseprice/sample_submission10.csv", index=False)



# =====================================================
import pandas as pd
import numpy as np

# NaN 포함
data = {
    '나이': np.random.randint(10, 60, size=10),
    '성적': np.random.choice(['상', '중', '하', np.nan], size=10)  # '상', '중', '하', NaN 중에서 랜덤 선택
}

df1 = pd.DataFrame(data)

df1 = pd.get_dummies(
    df1,
    columns= df1.select_dtypes(include=[object]).columns,
    drop_first=True
    )

# ------------------------------------------------------
# NaN 없음
data = {
    '나이': np.random.randint(10, 60, size=10),
    '성적': np.random.choice(['상', '중', '하'], size=10)  # '상', '중', '하', NaN 중에서 랜덤 선택
}

df2 = pd.DataFrame(data)

df2 = pd.get_dummies(
    df2,
    columns= df2.select_dtypes(include=[object]).columns,
    drop_first=True
    )
