import pandas as pd
import numpy as np


train = pd.read_csv('features_flat_base_train.csv')
test = pd.read_csv('features_flat_base_test.csv')


def merge(filename, on=['bulk_id', 'spalen', 'date1']):
  global train
  global test
  data = pd.read_csv(filename)
  train = pd.merge(left=train, right=data, on=on, how='left')
  test = pd.merge(left=test, right=data, on=on, how='left')
  
def drop(name):
  global train
  global test
  test = test.drop(name, axis=1)
  train = train.drop(name, axis=1)

for feature in ['features_flat_spalen.csv', 'features_salestart.csv', ]:
  pass

merge('features_flat_spalen.csv', on=['bulk_id', 'spalen'])

train['from_sale_start'] = train['date_salestart'] - train['date1']
test['from_sale_start'] = test['date_salestart'] - test['date1']

train['from_date_settle'] = train['date_settle'] - train['date1']
test['from_date_settle'] = test['date_settle'] - test['date1']


merge('features_salestart.csv')

drop('shadow_start_square')

merge('features_counter.csv')
merge('features_price.csv')
merge('features_status.csv')
merge('features_extra.csv', on=['date1'])

drop(['date_settle', 'ratio_vsre', 'num_flats'])

train.to_csv('FINAL_TRAIN.csv', index=False)
test.to_csv('FINAL_TEST.csv', index=False)


print(__file__)
