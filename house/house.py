import pandas as pd

house = pd.read_csv('house/train.csv')

house['SalePrice'].mean()

sub = pd.read_csv('house/sample_submission.csv')
sub['SalePrice'].mean()

sub['SalePrice'] = house['SalePrice'].mean()
sub

sub.to_csv('./house/sub.csv', index = False)
