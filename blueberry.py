import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
## 필요한 데이터 불러오기
house_train=pd.read_csv("blue/train.csv")
house_test=pd.read_csv("blue/test.csv")
sub_df=pd.read_csv("blue/sample_submission.csv")


house_train=house_train.iloc[:,1:]
house_test=house_test.iloc[:,1:]

house_train.shape
house_test.shape
train_n=len(house_train)


# 통합 df 만들기 + 더미코딩
df = pd.concat([house_train, house_test], ignore_index=True)

#df.info()

df.select_dtypes(include=[object]).columns

df = pd.get_dummies(
    df,
    columns= df.select_dtypes(include=[object]).columns,
    drop_first=True
    )
df

# train / test 데이터셋
train_df=df.iloc[:train_n,]
test_df=df.iloc[train_n:,]



## train
train_x=train_df.drop("yield",axis=1)
train_y=train_df["yield"]

## test
test_x=test_df.drop("yield",axis=1)
# 모든 열에 대해 1제곱부터 6제곱까지 계산하여 추가
for col in train_x.columns:
    for i in range(1, 6):  # 1제곱부터 6제곱까지
        train_x[f'{col}_pow_{i}'] = train_x[col] ** i
# 모든 열에 대해 1제곱부터 6제곱까지 계산하여 추가
for col in test_x.columns:
    for i in range(1, 6):  # 1제곱부터 6제곱까지
        test_x[f'{col}_pow_{i}'] = test_x[col] ** i


# 교차 검증 설정
kf = KFold(n_splits=5, shuffle=True, random_state=2024) #셔플한 값 만들기

# 알파 값 설정 처음에는 값간격을 크게하고 범위를 넓혀서 찾은후
# 세세한 값을 찾기 위해서 값간격을 작게하고 범위를 좁혀서 세세한 값을 찾는다
alpha_values = np.arange(0,10, 0.1)

# 각 알파 값에 대한 교차 검증 점수 저장
mean_scores = np.zeros(len(alpha_values))


def rmse(model):
    score = np.sqrt(-cross_val_score(model, train_x, train_y, cv = kf, n_jobs=-1,
                                     scoring = "neg_mean_squared_error").mean())
    return(score)



k=0
for alpha in alpha_values:
    ridge = Ridge(alpha=alpha)
    mean_scores[k] = rmse(ridge)
    k += 1

# 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

df

# 결과 시각화
plt.plot(df['lambda'], df['validation_error'], label='Validation Error', color='red')
plt.xlabel('Lambda')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Lasso Regression Train vs Validation Error')
plt.show()

# 최적의 alpha 값 찾기
optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)


#############
# 선형 회귀 모델 생성
model = Ridge(alpha=5)

# 모델 학습
model.fit(train_x, train_y)  # 자동으로 기울기, 절편 값을 구해줌



pred_y=model.predict(test_x) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
sub_df["yield"] = pred_y
sub_df
# csv 파일로 내보내기
sub_df.to_csv("./blue/all_Ridge_re_5.csv", index=False)

import numpy as np
import pandas as pd
