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
fig.show()

# 레이아웃 업데이트
fig.update_layout(
    title={
        'text': "<span style = 'color : blue; font-weight:bold;'> 팔머펭귄 </span>",
        }
)
fig

