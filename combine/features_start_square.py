import pandas as pd
import numpy as np


train = pd.read_csv('../files-pik_digital_day/train.csv', encoding='cp1251')
train['date1'] = pd.to_datetime(train['date1'])
train['date1'] = pd.DatetimeIndex(train['date1']).astype(np.int64)

test = pd.read_csv('../files-pik_digital_day/test.csv', encoding='cp1251')
test['date1'] = pd.to_datetime(test['date1'])
test['date1'] = pd.DatetimeIndex(test['date1']).astype(np.int64)

bulk_df = pd.read_csv('features_flat_spalen.csv')

start_sale = pd.read_csv('features_salestart.csv')

train = pd.merge(left=train, right=start_sale, on=['bulk_id', 'spalen', 'date1'], how='left')
test = pd.merge(left=test, right=start_sale, on=['bulk_id', 'spalen', 'date1'], how='left')

train = pd.merge(left=train, right=bulk_df, on=['bulk_id', 'spalen'], how='left')
test = pd.merge(left=test, right=bulk_df, on=['bulk_id', 'spalen'], how='left')

# GG
train1 = train[train['idle_square'] < 0.1]
train2 = train[train['sum_still_sale'] == train['start_square']]
train3 = train[train['shadow_start_square'] == train['start_square']]
train4 = train[['bulk_id', 'idle_square', 'sum_still_sale', 'shadow_start_square', 'start_square', 'square_sum']]
print(train1.shape, train2.shape, train3.shape)

# IT works ! or not ...
train['sq_eq'] = train['shadow_start_square']  # start_square
train.loc[train['shadow_start_square'] != train['start_square'], 'sq_eq'] = np.nan
train.loc[train['idle_square'] > 0, 'sq_eq'] = np.nan

# train['sq_eq'] = train['sum_still_sale']  # was sum_still_sale, start_square
# train.loc[train['idle_square'] > 0, 'sq_eq'] = np.nan

test['sq_eq'] = test['shadow_start_square']
test.loc[test['idle_square'] > 0, 'sq_eq'] = np.nan

#print(train1.shape)
#print(train2.shape)

super_tt = pd.concat([train[['date1', 'bulk_id', 'spalen', 'sq_eq']], test[['date1', 'bulk_id', 'spalen', 'sq_eq']]],
                     ignore_index=True)
super_tt = super_tt.drop_duplicates()

super_tt.to_csv('features_start_square.csv', index=False)

print(__file__)
