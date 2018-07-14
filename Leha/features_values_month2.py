import numpy as np
import pandas as pd


train = pd.read_csv('../files-pik_digital_day/train.csv', encoding='cp1251')

# semifinals
# q_periods = {
#     ('2015-06-01', '2015-08-01'): ['2015-08-01', '2015-09-01'],
#     ('2015-08-01', '2015-10-01'): ['2015-10-01', '2015-11-01'],
#     ('2015-10-01', '2015-12-01'): ['2015-12-01', '2016-01-01'],
#     ('2015-12-01', '2016-02-01'): ['2016-02-01', '2016-03-01'],
#     ('2016-02-01', '2016-04-01'): ['2016-04-01', '2016-05-01'],
#     ('2016-04-01', '2016-06-01'): ['2016-06-01', '2016-07-01'],
#     ('2016-06-01', '2016-08-01'): ['2016-08-01', '2016-09-01'],
#     ('2016-08-01', '2016-10-01'): ['2016-10-01', '2016-11-01'],
#     ('2016-10-01', '2016-12-01'): ['2016-12-01', '2017-01-01'],
#     ('2016-12-01', '2017-02-01'): ['2017-02-01', '2017-03-01'],
#     ('2017-02-01', '2017-04-01'): ['2017-04-01', '2017-05-01'],
#     ('2017-04-01', '2017-06-01'): ['2017-06-01', '2017-07-01'],
#     ('2017-06-01', '2017-08-01'): ['2017-08-01', '2017-09-01'],
#     ('2017-08-01', '2017-10-01'): ['2017-10-01', '2017-11-01'],
#     ('2017-10-01', '2017-12-01'): ['2017-12-01', '2018-01-01'],
#     ('2017-12-01', '2018-02-01'): ['2018-02-01', '2018-03-01'],
#     ('2018-02-01', '2018-04-01'): ['2018-04-01', '2018-05-01'],
#     ('2018-04-01', '2018-06-01'): ['2018-06-01', '2018-07-01'],
# }
q_periods = {
    ('2015-07-01', '2015-09-01'): ['2015-09-01', '2015-10-01'],
    ('2015-09-01', '2015-11-01'): ['2015-11-01', '2015-12-01'],
    ('2015-11-01', '2016-01-01'): ['2016-01-01', '2016-02-01'],
    ('2016-01-01', '2016-03-01'): ['2016-03-01', '2016-04-01'],
    ('2016-03-01', '2016-05-01'): ['2016-05-01', '2016-06-01'],
    ('2016-05-01', '2016-07-01'): ['2016-07-01', '2016-08-01'],
    ('2016-07-01', '2016-09-01'): ['2016-09-01', '2016-10-01'],
    ('2016-09-01', '2016-11-01'): ['2016-11-01', '2016-12-01'],
    ('2016-11-01', '2017-01-01'): ['2017-01-01', '2017-02-01'],
    ('2017-01-01', '2017-03-01'): ['2017-03-01', '2017-04-01'],
    ('2017-03-01', '2017-05-01'): ['2017-05-01', '2017-06-01'],
    ('2017-05-01', '2017-07-01'): ['2017-07-01', '2017-08-01'],
    ('2017-07-01', '2017-09-01'): ['2017-09-01', '2017-10-01'],
    ('2017-09-01', '2017-11-01'): ['2017-11-01', '2017-12-01'],
    ('2017-11-01', '2018-01-01'): ['2018-01-01', '2018-02-01'],
    ('2018-01-01', '2018-03-01'): ['2018-03-01', '2018-04-01'],
    ('2018-03-01', '2018-05-01'): ['2018-05-01', '2018-06-01'],
}

train['date1'] = pd.to_datetime(train['date1'])
train['date1'] = pd.DatetimeIndex(train['date1']).astype(np.int64)


what_we_want = train[['bulk_id', 'spalen']].drop_duplicates()
ff = None

for d1, d123 in q_periods.items():
    for dd in d123:
        new_one = what_we_want.copy()
        new_one['date1'] = pd.DatetimeIndex([dd]).astype(np.int64)[0]

        if ff is None:
            ff = new_one.copy()
        else:
            ff = pd.concat([ff, new_one], ignore_index=True)

ff['start_square_q3'] = 0
ff['value_q3'] = 0
# ff['value_diff_q3'] = 0
ff['price_q3'] = 0
# 'plan_s', 'plan_m', 'plan_l', 'vid_0', 'vid_1', 'vid_2'
# ff['plan_s'] = 0


for d1, d123 in q_periods.items():
    start_p = pd.DatetimeIndex([d1[0]]).astype(np.int64)[0]
    end_p = pd.DatetimeIndex([d1[1]]).astype(np.int64)[0]

    value_q3 = train[(train['date1'] >= start_p) & (train['date1'] < end_p)]
    value_q3 = value_q3.groupby(['bulk_id', 'spalen'], as_index=False).agg({'value': np.sum})
    value_q3 = value_q3.rename(columns={'value': 'value_q3_2'})

    # value_diff_q3 = train[(train['date1'] >= start_p) & (train['date1'] < end_p)]
    # value_diff_q3 = value_diff_q3.groupby(['bulk_id', 'spalen'], as_index=False).agg({'value': lambda s: s.values[-1] - s.values[0]})
    # value_diff_q3 = value_diff_q3.rename(columns={'start_square': 'value_diff_q3_2'})

    start_square_q3 = train[(train['date1'] >= start_p) & (train['date1'] < end_p)]
    start_square_q3 = start_square_q3.groupby(['bulk_id', 'spalen'], as_index=False).agg({'start_square': np.sum})
    start_square_q3 = start_square_q3.rename(columns={'start_square': 'start_square_q3_2'})
    #
    price_q3 = train[(train['date1'] >= start_p) & (train['date1'] < end_p)]
    price_q3 = price_q3.groupby(['bulk_id', 'spalen'], as_index=False).agg({'price': lambda s: s.values[-1] - s.values[0]})
    price_q3 = price_q3.rename(columns={'price': 'price_q3_2'})

    # plan_s = train[(train['date1'] >= start_p) & (train['date1'] < end_p)]
    # plan_s = plan_s.groupby(['bulk_id', 'spalen'], as_index=False).agg({'plan_s': np.sum})
    # plan_s = plan_s.rename(columns={'plan_s': 'plan_s2'})

    for dd in d123:
        value_q3['date1'] = pd.DatetimeIndex([dd]).astype(np.int64)[0]
        ff = pd.merge(left=ff, right=value_q3, on=['bulk_id', 'spalen', 'date1'], how='left')
        ff = ff.fillna(0)
        ff['value_q3'] = ff['value_q3'] + ff['value_q3_2']
        ff = ff.drop(['value_q3_2'], axis=1)

        start_square_q3['date1'] = pd.DatetimeIndex([dd]).astype(np.int64)[0]
        ff = pd.merge(left=ff, right=start_square_q3, on=['bulk_id', 'spalen', 'date1'], how='left')
        ff = ff.fillna(0)
        ff['start_square_q3'] = ff['start_square_q3'] + ff['start_square_q3_2']
        ff = ff.drop(['start_square_q3_2'], axis=1)
        #
        price_q3['date1'] = pd.DatetimeIndex([dd]).astype(np.int64)[0]
        ff = pd.merge(left=ff, right=price_q3, on=['bulk_id', 'spalen', 'date1'], how='left')
        ff = ff.fillna(0)
        ff['price_q3'] = ff['price_q3'] + ff['price_q3_2']
        ff = ff.drop(['price_q3_2'], axis=1)

        # value_diff_q3['date1'] = pd.DatetimeIndex([dd]).astype(np.int64)[0]
        # ff = pd.merge(left=ff, right=value_diff_q3, on=['bulk_id', 'spalen', 'date1'], how='left')
        # ff = ff.fillna(0)
        # ff['value_diff_q3'] = ff['value_diff_q3'] + ff['value_diff_q3_2']
        # ff = ff.drop(['value_diff_q3_2'], axis=1)

        # plan_s['date1'] = pd.DatetimeIndex([dd]).astype(np.int64)[0]
        # ff = pd.merge(left=ff, right=plan_s, on=['bulk_id', 'spalen', 'date1'], how='left')
        # ff = ff.fillna(0)
        # ff['plan_s'] = ff['plan_s'] + ff['plan_s2']
        # ff = ff.drop(['plan_s2'], axis=1)

ff = ff.rename(columns={'value_q3': 'mm_value_q3', 'start_square_q3': 'mm_start_square_q3', 'price_q3': 'mm_price_q3'})
ff = ff.drop(['mm_price_q3'], axis=1)
# ff = ff.rename(columns={'value_q3': 'mm_value_q3'})

ff = ff.replace(0, np.nan)
ff.to_csv('features_values_month2.csv', index=False)

print(__file__)
