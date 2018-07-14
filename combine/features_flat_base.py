import pandas as pd
import numpy as np


train = pd.read_csv('../files-pik_digital_day/train.csv', encoding='cp1251')
# start_square_avg = train.groupby(['bulk_id', 'spalen'], as_index=False).agg({'start_square': np.mean})
train = train.drop(['id', 'start_square', 'plan_s', 'plan_m', 'plan_l', 'vid_0', 'vid_1', 'vid_2'], axis=1)
test = pd.read_csv('../files-pik_digital_day/test.csv', encoding='cp1251')

train['date1'] = pd.to_datetime(train['date1'])
train['date1'] = pd.DatetimeIndex(train['date1']).astype(np.int64)
test['date1'] = pd.to_datetime(test['date1'])
test['date1'] = pd.DatetimeIndex(test['date1']).astype(np.int64)

train['price_m'] = train['price'] / train['mean_sq']
test['price_m'] = test['price'] / test['mean_sq']

# train['flat2car'] = train['Количество помещений'] / train['Машиномест']
# test['flat2car'] = test['Количество помещений'] / test['Машиномест']
train['car_sum_distance'] = (train['До большой дороги на машине(км)']+train['До удобной авторазвязки на машине(км)'])
test['car_sum_distance'] = (test['До большой дороги на машине(км)']+test['До удобной авторазвязки на машине(км)'])


# train['price_mm'] = train['price'] * train['mean_sq']
# test['price_mm'] = test['price'] * test['mean_sq']
# train['metro_importance'] = train['До метро пешком(км)'] / (train['Машиномест']+1)
# test['metro_importance'] = test['До метро пешком(км)'] / (test['Машиномест']+1)
# train['metro_importance'] = train['До метро пешком(км)'] / (train['До большой дороги на машине(км)']+train['До удобной авторазвязки на машине(км)'])
# test['metro_importance'] = test['До метро пешком(км)'] / (test['До большой дороги на машине(км)']+test['До удобной авторазвязки на машине(км)'])
# train['metro_importance'] = train['До метро пешком(км)'] / (train['До большой дороги на машине(км)'] + 1)
# test['metro_importance'] = test['До метро пешком(км)'] / (test['До большой дороги на машине(км)'] + 1)

# avg_price_month_train = train.groupby(['date1'], as_index=False).agg({'price': np.mean})
# avg_price_month_test = test.groupby(['date1'], as_index=False).agg({'price': np.mean})
# avg_price_month_train = avg_price_month_train.rename(columns={'price': 'avg_p_m'})
# avg_price_month_test = avg_price_month_test.rename(columns={'price': 'avg_p_m'})
# train = pd.merge(left=train, right=avg_price_month_train, on=['date1'], how='left')
# test = pd.merge(left=test, right=avg_price_month_test, on=['date1'], how='left')
# train['avg_p_m'] = train['price_m'] / train['avg_p_m']
# test['avg_p_m'] = test['price_m'] / test['avg_p_m']

# bulk_id_train = train.groupby(['bulk_id'], as_index=False).agg({'spalen': len})
# bulk_id_train = bulk_id_train.rename(columns={'spalen': 'just_len'})
# train = pd.merge(left=train, right=bulk_id_train, on=['bulk_id'], how='left')
#
# bulk_id_test = test.groupby(['bulk_id'], as_index=False).agg({'spalen': len})
# bulk_id_test = bulk_id_test.rename(columns={'spalen': 'just_len'})
# test = pd.merge(left=test, right=bulk_id_test, on=['bulk_id'], how='left')
# train

# price_change = train.groupby(['bulk_id', 'spalen'], as_index=False).agg({'price': lambda s: s.values[-1] - s.values[0]})  # np.std
# price_change = train.groupby(['bulk_id', 'spalen'], as_index=False).agg({'price': lambda s: s.values[-1] - s.values[0]})  # np.std
# price_change = price_change.rename(columns={'price': 'price_change'})
# train = pd.merge(left=train, right=price_change, on=['bulk_id', 'spalen'], how='left')
# #
# price_change = test.groupby(['bulk_id', 'spalen'], as_index=False).agg({'price': lambda s: s.values[-1] - s.values[0]})  # np.max(s) - np.min(s)
# price_change = price_change.rename(columns={'price': 'price_change'})
# test = pd.merge(left=test, right=price_change, on=['bulk_id', 'spalen'], how='left')

# end_num = test.groupby(['bulk_id', 'spalen'], as_index=False).agg({'date1': np.sum})
# end_num = end_num.rename(columns={'date1': 'num_in_last3'})
# train = pd.merge(left=train, right=end_num, on=['bulk_id', 'spalen'], how='left')
# test = pd.merge(left=test, right=end_num, on=['bulk_id', 'spalen'], how='left')

# features_flat_base = pd.read_csv('features_flat_base.csv')
# train = pd.merge(left=train, right=features_flat_base, on=['bulk_id'], how='left')
# test = pd.merge(left=test, right=features_flat_base, on=['bulk_id'], how='left')

train.to_csv('features_flat_base_train.csv', index=False)
test.to_csv('features_flat_base_test.csv', index=False)

print(__file__)
