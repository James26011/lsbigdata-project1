import pandas as pd

house = pd.read_csv('house/train.csv')
house
house['SalePrice'].mean()

sub = pd.read_csv('house/sample_submission.csv')
sub['SalePrice'].mean()

sub['SalePrice'] = house['SalePrice'].mean()
sub

sub.to_csv('./house/sub.csv', index = False)




년도 평균을 구해서
테스트 set의 집값을 예측

21세기 전후 / 전이 더 싸다

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

# ==============이삭 ver============================
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

# =================================================
