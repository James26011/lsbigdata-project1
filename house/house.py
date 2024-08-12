import pandas as pd

house = pd.read_csv('house/train.csv')

house

house['SalePrice'].mean()

sub = pd.read_csv('house/sample_submission.csv')
sub['SalePrice'].mean()

sub['SalePrice'] = house['SalePrice'].mean()
sub

sub.to_csv('./house/sub.csv', index = False)



# 연도 범위, 평균 확인
house['YearBuilt'].describe()

# 년도 별 그룹바이, 가격 평균
new = house.groupby('YearBuilt',as_index = False) \
           .agg(new_price = ('SalePrice','mean'))
new

# test 불러오기
test = pd.read_csv('house/test.csv')

# test랑 가격 평균 합체
new2 = pd.merge(test, new, how = 'left', on = 'YearBuilt')
new2.to_csv('./house/new2.csv', index = False)
new2

pd.isna(new2).sum()

#new price 결측치 전체 평균값넣기
new2 = new2.fillna(new2['new_price'].mean())
new2
# 제출용데이터 불러오기
sub = pd.read_csv('house/sample_submission.csv')
sub2 = sub.copy()
#제출용 데이터에 년도별 그룹합치기
sub2['SalePrice'] = new2['new_price']
sub2.to_csv('./house/sub2.csv', index = False)

# ==============이삭T ver============================
house_train = pd.read_csv('house/train.csv')
house_train = house_train[['Id','YearBuilt','SalePrice']]
house_train.info()

# 연도별 평균
house_mean = house_train.groupby('YearBuilt') \ 
                        .agg(mean_year = ('SalePrice','mean'))
house_mean           

house_test = pd.read_csv('house/test.csv')
house_test = house_test[['Id','YearBuilt']]
house_test

house_test = pd.merge(house_test, house_mean, how = 'left', on = 'YearBuilt')
house_test
house_test = house_test.rename(columns = {'mean_year' : 'SalePrice'})
house_test

# 비어있는 집 확인
house_test['SalePrice'].isna().sum()
house_test.loc[house_test['SalePrice'].isna()]

# 집 값 채우기
house_mean = house_train['SalePrice'].mean()
house_test['SalePrice'] = house_test['SalePrice'].fillna(house_mean)

# sub 데이터 불러오기
sub_df = pd.read_csv('house/sample_submission.csv')
sub_df

# 집 가격 바꾸기
sub_df['SalePrice'] = house_test['SalePrice']
sub_df

sub_df.to_csv('./house/sub2.csv', index = False)

# =================시각화================================
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

house_train = pd.read_csv('house/train.csv')

# LotArea(대지면적) -TotRmsAbvGrd(지상층 방개수)
house_train = house_train[['LotArea','TotRmsAbvGrd','SalePrice']]

house_train['LotArea'].isna().sum()
house_train['LotArea'].describe()

house_train['TotRmsAbvGrd'].isna().sum()
house_train['TotRmsAbvGrd'].describe() #최소값 2, 최대값 14
house_train['TotRmsAbvGrd'].value_counts().sort_index()

house_train['SalePrice'].isna().sum()



# 데이터 준비
tot = house_train.groupby('TotRmsAbvGrd') \
           .agg(mean_price=('SalePrice', 'mean'),
                mean_area=('LotArea', 'mean'))

# 그래프 그리기
fig, ax1 = plt.subplots()
# 첫 번째 y축 (mean_price)
sns.lineplot(data=tot, x='TotRmsAbvGrd', y='mean_price', color='black', linestyle='-', ax=ax1)
ax1.set_xlabel('TotRmsAbvGrd')
ax1.set_ylabel('Mean_price', color='black')
ax1.tick_params(axis='y', labelcolor='black')


# 두 번째 y축 (mean_area)
ax2 = ax1.twinx()  # 공유 x축을 가지는 두 번째 y축
sns.lineplot(data=tot, x='TotRmsAbvGrd', y='mean_area', color='red', linestyle='-', ax=ax2)
ax2.set_ylabel('Mean Area', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# 그래프 제목 및 레이아웃 조정
plt.title('Mean_price and Mean Area by Total Rooms Above Ground')
fig.tight_layout()

# 그래프 표시
plt.show()
plt.clf()

house_train['TotRmsAbvGrd'].describe() #최소값 2, 최대값 14
house_train['TotRmsAbvGrd'].value_counts().sort_index()




# ============================================


new_house = pd.read_csv('./house/houseprice-with-lonlat.csv')
new_house.columns
