import numpy as np
import matplotlib.pyplot as plt
import json

geo_seoul = json.load(open("./data/bigfiles/SIG_Seoul.geojson", encoding="UTF-8"))

# 데이터 탐색
type(geo_seoul)
len(geo_seoul)
geo_seoul.keys()
geo_seoul["features"][0]
len(geo_seoul["features"])
len(geo_seoul["features"][0])
geo_seoul["features"][0].keys()

# 숫자가 바뀌면 "구"가 바뀌는구나!
geo_seoul["features"][2]["properties"]
geo_seoul["features"][0]["geometry"]

# 리스트로 정보 빼오기
coordinate_list=geo_seoul["features"][2]["geometry"]["coordinates"]
len(coordinate_list[0][0])
coordinate_list[0][0]

coordinate_array=np.array(coordinate_list[0][0])
x=coordinate_array[:,0]
y=coordinate_array[:,1]

plt.plot(x, y)
plt.show()
plt.clf()

# 함수로 만들기
def draw_seoul(num):
    gu_name=geo_seoul["features"][num]["properties"]["SIG_KOR_NM"]
    coordinate_list=geo_seoul["features"][num]["geometry"]["coordinates"]
    coordinate_array=np.array(coordinate_list[0][0])
    x=coordinate_array[:,0]
    y=coordinate_array[:,1]

    plt.rcParams.update({"font.family": "Malgun Gothic"})
    plt.plot(x, y)
    plt.title(gu_name)
    plt.show()
    plt.clf()
    
    return None

draw_seoul(12)

# ===================================
import seaborn as sns
import pandas as pd

def df_gu(x):
    import numpy as np
    import pandas as pd
    coordinate_list = geo_seoul["features"][x]["geometry"]["coordinates"][0][0]
    coordinate_array = np.array(coordinate_list)
    df = pd.DataFrame({})
    df["gu_name"] = [geo_seoul["features"][x]["properties"]["SIG_KOR_NM"]]*len(coordinate_array)
    df["x"] = coordinate_array[:,0]
    df["y"] = coordinate_array[:,1]
    return df
df_gu(0)

result = pd.DataFrame({})

for x in range(len(geo_seoul["features"])):
    result = pd.concat([result,df_gu(x)])
    
result = result.reset_index(drop=True)
result

# 1 
sns.lineplot(data = result,x = 'x', y = 'y', hue= "gu_name")
plt.legend(fontsize = 2)
plt.show()
plt.clf()

# 구이름 만들기
# 방법 1
gu_name=list()
for i in range(25):
    gu_name.append(geo_seoul["features"][i]["properties"]["SIG_KOR_NM"])
gu_name

# 방법 2
gu_name = [geo_seoul["features"][i]["properties"]["SIG_KOR_NM"] for i in range(25))]
gu_name

# x, y 판다스 데이터 프레임
import pandas as pd

def make_seouldf(num):
    gu_name=geo_seoul["features"][num]["properties"]["SIG_KOR_NM"]
    coordinate_list=geo_seoul["features"][num]["geometry"]["coordinates"]
    coordinate_array=np.array(coordinate_list[0][0])
    x=coordinate_array[:,0]
    y=coordinate_array[:,1]

    return pd.DataFrame({"gu_name":gu_name, "x": x, "y": y})

make_seouldf(1)

result=pd.DataFrame({})
for i in range(25):
    result=pd.concat([result, make_seouldf(i)], ignore_index=True)    

result

sns.scatterplot(data = result,
                x='x', y = 'y', hue = 'gu_name', s=1, legend = False)
plt.rcParams.update({"font.family": "Malgun Gothic"})
plt.show()
plt.clf()

# # 데이터프레임 concat 예제
# df_a = pd.DataFrame({
#     'ID': [],
#     'Name': [],
#     'Age': []
# })
# 
# df_b = pd.DataFrame({
#     'ID': [4, 5, 6],
#     'Name': ['David', 'Eva', 'Frank'],
#     'Age': [40, 45, 50]
# })
# df_a=pd.concat([df_a, df_b])

# 서울 그래프 그리기
result
gangnam_df = result.assign(is_gangnam = np.where(result['gu_name'] == '강남구','강남','안강남'))
sns.scatterplot(
    data = gangnam_df,
    x = 'x', y = 'y',
    hue = 'is_gangnam',
    palette = 'deep', s=2)
plt.show()
plt.clf()

# ===============================

import json
geo_seoul = json.load(open('./data/bigfiles/SIG_Seoul.geojson', encoding = 'UTF-8'))
geo_seoul['features'][0]['properties']

df_pop = pd.read_csv('./data/bigfiles/Population_SIG.csv')
df_pop.head()

df_seoulpop = df_pop.iloc[1:26]
df_seoulpop['code'] = df_seoulpop['code'].astype(str)
df_seoulpop.info()

# !pip install folium
import folium

center_x = result['x'].mean()
center_y = result['y'].mean()
my_map = folium.Map(location = [37.551, 126.97], zoom_start = 5, tiles = 'cartodbpositron')
# my_map.save('map_seoul.html')

# 코로플릿
map_sig = folium.Map(locaion = [3.95,127.7],
                     zoom_start = 8,
                     tiles = 'cartodbpositron')

folium.Choropleth(geo_data = geo_seoul,
                  data = df_seoulpop,
                  columns = ('code','pop'),
                  key_on = 'feature.properties.SIG_CD') \
                    .add_to(map_sig)
                    
bins = list(df_seoulpop['pop'].quantile([0,0.2,0.4,0.6,0.8,1]))
bins

my_map = folium.Choropleth(
    geo_data = geo_seoul,
    data = df_seoulpop,
    columns = ('code','pop'),
    key_on = 'feature.properties.SIG_CD',
    fill_color = 'YlGnBu',
    fill_opacity = 0.5,
    line_opacity = 0.5,
    bins = bins) \
        .add_to(map_sig)
        
my_map.save('map_seoul.html')

# 점 찍는 법
# make_seouldf(0).iloc[:,1:3].mean()
folium.Marker([37.583744, 126.983800], popup="종로구").add_to(map_sig)
map_sig.save("map_seoul.html")

geo_seoul["features"][5]["properties"]["SIG_KOR_NM"]

gu_name = list()
for i in range(len(geo_seoul["features"])):
    gu_name.append(geo_seoul["features"][i]["properties"]["SIG_KOR_NM"])
    
gu_name



for i in range(25):
    means = make_seouldf(i)[['x', 'y']].mean()  # 종로구 중앙
    folium.Marker([means[1], means[0]], popup = gu_name[i]).add_to(map_sig)
map_sig.save("map_seoul.html")
    
