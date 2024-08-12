# 데이터 패키지 설치
# pip install palmerpenguins
import pandas as pd
import numpy as np
import plotly.express as px
from palmerpenguins import load_penguins

penguins = load_penguins()
penguins.head()

# x: bill_length_mm
# y: bill_depth_mm  

# 산점도 생성
fig = px.scatter(
    penguins,
    x="bill_length_mm",
    y="bill_depth_mm",
    color="species",
    # trendline = 'ols'
)

# 레이아웃 업데이트
fig.update_layout(
    title=dict(
        text="팔머펭귄 종별 부리 길이 vs. 깊이",
        font=dict(color="white", size=35)  # 제목 폰트 크기 조정
    ),
    paper_bgcolor="black",
    plot_bgcolor="black",
    font=dict(color="white"),
    xaxis=dict(
        title=dict(text="부리 길이 (mm)", font=dict(color="white")), 
        tickfont=dict(color="white"),
        gridcolor='rgba(255, 255, 255, 0.2)'  # 그리드 색깔 조정
    ),
    yaxis=dict(
        title=dict(text="부리 깊이 (mm)", font=dict(color="white")), 
        tickfont=dict(color="white"),
        gridcolor='rgba(255, 255, 255, 0.2)'  # 그리드 색깔 조정
    ),
    legend=dict(
        title=dict(text="펭귄 종", font=dict(color="white")),  # 범례 제목 변경
        font=dict(color="white")
    ),
)

# 점 크기 조정
fig.update_traces(marker=dict(size=12,opacity=0.6))  # 점 크기 조정 (원하는 크기로 변경 가능)

fig.show()

# =====================================

from sklearn.linear_model import LinearRegression
model = LinearRegression()

penguins = penguins.dropna()

x = penguins[['bill_length_mm']]
y = penguins['bill_depth_mm']

model.fit(x,y)
linear_fit = model.predict(x)
model.coef_

fig.add_trace(
    go.Scatter(
        mode = 'lines',
        x = penguins['bill_length_mm'], y = linear_fit,
        name = '선형회귀직선',
        line = dict(dash = 'dot',color = 'white')
    )
)
fig.show()

# =============================================================
# 범주형 변수로 회귀분석 진행하기
# 범주형 변수인 'species'를 더미 변수로 변환
penguins
penguins_dummies = pd.get_dummies(penguins, 
                                  columns=['species'],
                                  drop_first=False)
penguins_dummies.columns
penguins_dummies.iloc[:,-3:]

# x와 y 설정
x = penguins_dummies[["bill_length_mm", "species_Chinstrap", "species_Gentoo"]]
y = penguins_dummies["bill_depth_mm"]

# 모델 학습
model = LinearRegression()
model.fit(x, y)

model.coef_
model.intercept_

regline_y = model.predict(x)
import matplotlib.pyplot as plt
plt.scatter(x['bill_length_mm'], regline_y, s=1)
plt.scatter(x['bill_length_mm'], y, color = 'black', s=1)
plt.show()
plt.clf()


# y = 0.20044313 * bill_length - 1.93 * species_Chinstrap - 5.1 * species_Gentoo + 10.56


