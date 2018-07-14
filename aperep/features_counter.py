import pandas as pd
import numpy as np
from common import date2int, inputdir


train = pd.read_csv(inputdir+'train.csv', encoding='cp1251')
test = pd.read_csv(inputdir+'test.csv', encoding='cp1251')

train['date1'] = date2int(train['date1'])
test['date1'] = date2int(test['date1'])

super_tt = pd.concat([train[['date1', 'bulk_id', 'spalen']], test[['date1', 'bulk_id', 'spalen']]], ignore_index=True)
super_tt = super_tt.drop_duplicates()

super_tt = super_tt.sort_values(by=['date1', 'bulk_id', 'spalen'])


months_time = [
    '2015-07-01',
    '2015-08-01',
    '2015-09-01',
    '2015-10-01',
    '2015-11-01',
    '2015-12-01',
    '2016-01-01',
    '2016-02-01',
    '2016-03-01',
    '2016-04-01',
    '2016-05-01',
    '2016-06-01',
    '2016-07-01',
    '2016-08-01',
    '2016-09-01',
    '2016-10-01',
    '2016-11-01',
    '2016-12-01',
    '2017-01-01',
    '2017-02-01',
    '2017-03-01',
    '2017-04-01',
    '2017-05-01',
    '2017-06-01',
    '2017-07-01',
    '2017-08-01',
    '2017-09-01',
    '2017-10-01',
    '2017-11-01',
    '2017-12-01',
    '2018-01-01',
    '2018-02-01',
    '2018-03-01',
    '2018-04-01',
    '2018-05-01'
]
months_time2 = []
for m in months_time:
    months_time2.append(pd.DatetimeIndex([m]).astype(np.int64)[0])

months_time = months_time2
del months_time2
# print(months_time)

def get_month_time(s1, minus_month=0):
    if s1 in months_time:
        ind1 = months_time.index(s1)
        ind2 = ind1 + minus_month
        if ind2 >= 0:
            return months_time[ind2]
    return None


def ff(x):
    bulk_id = x['bulk_id'].values[0]
    spalen = x['spalen'].values[0]
    date1 = x['date1'].values[0]

    d1 = x['date1'].values[0]
    ind1 = get_month_time(d1, 0)

    super_tt_filtered = super_tt[(super_tt['bulk_id'] == bulk_id) & (super_tt['spalen'] == spalen)]
    month_from_first = super_tt_filtered[super_tt_filtered['date1'] <= ind1].shape[0]

    # super_tt_filtered = super_tt[(super_tt['bulk_id'] == bulk_id) & (super_tt['spalen'] == spalen)]
    # month_from_first = d1 - np.min(super_tt_filtered['date1'].values)

    return pd.Series([bulk_id, spalen, date1, month_from_first], index=['bulk_id', 'spalen', 'date1', 'month_from_first'])


super_tt2 = super_tt.groupby(['bulk_id', 'spalen', 'date1'], as_index=False).apply(ff)
super_tt2.to_csv('features_counter.csv', index=False)

print(__file__)
