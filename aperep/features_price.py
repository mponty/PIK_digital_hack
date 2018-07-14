import pandas as pd
import numpy as np
from common import date2int, inputdir, unique_print


train = pd.read_csv(inputdir+'train.csv', encoding='cp1251')
test = pd.read_csv(inputdir+'test.csv', encoding='cp1251')

train['date1'] = date2int(train['date1'])
test['date1'] = date2int(test['date1'])

super_tt = pd.concat([train[['date1', 'bulk_id', 'spalen', 'price']], test[['date1', 'bulk_id', 'spalen', 'price']]], ignore_index=True)
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
    unique_print(bulk_id)

    d1 = x['date1'].values[0]
    month_before = get_month_time(d1, -1)

    # super_tt_filtered = super_tt[(super_tt['bulk_id'] == bulk_id) & (super_tt['spalen'] == spalen) & (super_tt['date1'] == date1)]
    # price_bulk_mean = np.mean(super_tt_filtered['price'].values)
    price_now = x['price'].values[0]

    if month_before is not None:
        # super_tt_filtered = super_tt[(super_tt['bulk_id'] == bulk_id) & (super_tt['spalen'] == spalen) & (super_tt['date1'] == month_before)]
        super_tt_filtered = super_tt[(super_tt['bulk_id'] == bulk_id) & (super_tt['spalen'] == spalen) & (super_tt['date1'] <= date1)]
        if super_tt_filtered.shape[0] > 0:
            # price_before = super_tt_filtered['price'].values[0]
            price_before = super_tt_filtered['price'].values[0]
            price_rel_change = price_now - price_before
            # price_rel_change = price_before / price_now
        else:
            price_rel_change = np.nan
    else:
        price_rel_change = np.nan

    just_bulk_len = super_tt[(super_tt['bulk_id'] == bulk_id)].shape[0]

    # return pd.Series([bulk_id, spalen, date1, price_bulk_mean], index=['bulk_id', 'spalen', 'date1', 'price_bulk_mean'])
    return pd.Series([bulk_id, spalen, date1, price_rel_change, just_bulk_len], index=['bulk_id', 'spalen', 'date1', 'price_rel_change', 'just_bulk_len'])


super_tt2 = super_tt.groupby(['bulk_id', 'spalen', 'date1'], as_index=False).apply(ff)
super_tt2.to_csv('features_price.csv', index=False)

print(__file__)
