import pandas as pd
import numpy as np


train = pd.read_csv('features_flat_base_train.csv')
test = pd.read_csv('features_flat_base_test.csv')


features_flat_base = pd.read_csv('features_flat_spalen.csv')
train = pd.merge(left=train, right=features_flat_base, on=['bulk_id', 'spalen'], how='left')
test = pd.merge(left=test, right=features_flat_base, on=['bulk_id', 'spalen'], how='left')

train['from_sale_start'] = train['date_salestart'] - train['date1']
test['from_sale_start'] = test['date_salestart'] - test['date1']

train['from_date_settle'] = train['date_settle'] - train['date1']
test['from_date_settle'] = test['date_settle'] - test['date1']

# if CROSS_VALIDATION:
#     features_flat_time = pd.read_csv('features_flat_time_cv.csv')
# else:
#     features_flat_time = pd.read_csv('features_flat_time.csv')
#
# train = pd.merge(left=train, right=features_flat_time, on=['bulk_id', 'spalen'], how='left')
# test = pd.merge(left=test, right=features_flat_time, on=['bulk_id', 'spalen'], how='left')

# works locally!!!!
# train['avg_sale_date2'] = train['avg_sale_date'] - train['date1']
# test['avg_sale_date2'] = test['avg_sale_date'] - test['date1']

features_salestart = pd.read_csv('features_salestart.csv')
train = pd.merge(left=train, right=features_salestart, on=['bulk_id', 'spalen', 'date1'], how='left')
test = pd.merge(left=test, right=features_salestart, on=['bulk_id', 'spalen', 'date1'], how='left')

# features_start_square = pd.read_csv('features_start_square.csv')
# train = pd.merge(left=train, right=features_start_square, on=['bulk_id', 'spalen', 'date1'], how='left')
# test = pd.merge(left=test, right=features_start_square, on=['bulk_id', 'spalen', 'date1'], how='left')
test = test.drop('shadow_start_square', axis=1)
train = train.drop('shadow_start_square', axis=1)

features_counter = pd.read_csv('features_counter.csv')
train = pd.merge(left=train, right=features_counter, on=['bulk_id', 'spalen', 'date1'], how='left')
test = pd.merge(left=test, right=features_counter, on=['bulk_id', 'spalen', 'date1'], how='left')


features_price = pd.read_csv('features_price.csv')
train = pd.merge(left=train, right=features_price, on=['bulk_id', 'spalen', 'date1'], how='left')
test = pd.merge(left=test, right=features_price, on=['bulk_id', 'spalen', 'date1'], how='left')


features_status3 = pd.read_csv('features_status_month3.csv')
train = pd.merge(left=train, right=features_status3, on=['bulk_id', 'spalen', 'date1'], how='left')
test = pd.merge(left=test, right=features_status3, on=['bulk_id', 'spalen', 'date1'], how='left')

# features_status2 = pd.read_csv('features_status2.csv')
# train = pd.merge(left=train, right=features_status2, on=['bulk_id', 'spalen', 'date1'], how='left')
# test = pd.merge(left=test, right=features_status2, on=['bulk_id', 'spalen', 'date1'], how='left')

features_extra = pd.read_csv('features_extra.csv')
train = pd.merge(left=train, right=features_extra, on=['date1'], how='left')
test = pd.merge(left=test, right=features_extra, on=['date1'], how='left')


features_values3 = pd.read_csv('features_values_month3.csv')
train = pd.merge(left=train, right=features_values3, on=['bulk_id', 'spalen', 'date1'], how='left')
test = pd.merge(left=test, right=features_values3, on=['bulk_id', 'spalen', 'date1'], how='left')

# features_values2 = pd.read_csv('features_values2.csv')
# train = pd.merge(left=train, right=features_values2, on=['bulk_id', 'spalen', 'date1'], how='left')
# test = pd.merge(left=test, right=features_values2, on=['bulk_id', 'spalen', 'date1'], how='left')

# features_train_test = pd.read_csv('features_train_test.csv')
# train = pd.merge(left=train, right=features_train_test, on=['bulk_id', 'spalen', 'date1'], how='left')
# test = pd.merge(left=test, right=features_train_test, on=['bulk_id', 'spalen', 'date1'], how='left')

# status_features = pd.read_csv('status_features3_cv.csv')
# train = pd.merge(left=train, right=status_features, on=['bulk_id', 'spalen'], how='left')
# test = pd.merge(left=test, right=status_features, on=['bulk_id', 'spalen'], how='left')

# test.loc[test['date1'] == 1519862400000000000, '100000003'] = None
# test.loc[test['date1'] == 1522540800000000000, '100000003'] = None
# train.loc[train['date1'] < pd.DatetimeIndex(['2017-07-01']).astype(np.int64)[0], '100000003'] = None
# train.loc[train['date1'] == pd.DatetimeIndex(['2017-12-01']).astype(np.int64)[0], '100000003'] = None
# train.loc[train['date1'] == pd.DatetimeIndex(['2018-01-01']).astype(np.int64)[0], '100000003'] = None

# train = train.drop(['date_settle'], axis=1)
# test = test.drop(['date_settle'], axis=1)
# train = train.drop(['date_settle', 'ratio_vsre', 'avg_sale_date', 'ratio_vre_16'], axis=1)
# test = test.drop(['date_settle', 'ratio_vsre', 'avg_sale_date', 'ratio_vre_16'], axis=1)

# train = train.drop(['date_settle', 'ratio_vsre', 'num_flats', 'price_rel_change'], axis=1)
# test = test.drop(['date_settle', 'ratio_vsre', 'num_flats', 'price_rel_change'], axis=1)

train = train.drop(['date_settle', 'ratio_vsre', 'flat_still_sale', 'sum_still_sale'], axis=1)
test = test.drop(['date_settle', 'ratio_vsre', 'flat_still_sale', 'sum_still_sale'], axis=1)
# train = train.drop(['date_settle', 'ratio_vsre'], axis=1)
# test = test.drop(['date_settle', 'ratio_vsre'], axis=1)


train.to_csv('FINAL_TRAIN_month3.csv', index=False)
test.to_csv('FINAL_TEST_month3.csv', index=False)


print(__file__)
