import pandas as pd
import numpy as np
from common import inputdir

train = pd.read_csv(inputdir+'train.csv', encoding='cp1251')
# start_square_avg = train.groupby(['bulk_id', 'spalen'], as_index=False).agg({'start_square': np.mean})
train = train.drop(['id', 'start_square', 'plan_s', 'plan_m', 'plan_l', 'vid_0', 'vid_1', 'vid_2'], axis=1)
test = pd.read_csv(inputdir+'test.csv', encoding='cp1251')

train['date1'] = pd.to_datetime(train['date1'])
train['date1'] = pd.DatetimeIndex(train['date1']).astype(np.int64)
test['date1'] = pd.to_datetime(test['date1'])
test['date1'] = pd.DatetimeIndex(test['date1']).astype(np.int64)

train['price_m'] = train['price'] / train['mean_sq']
test['price_m'] = test['price'] / test['mean_sq']

train.to_csv('features_flat_base_train.csv', index=False)
test.to_csv('features_flat_base_test.csv', index=False)

print(__file__)
