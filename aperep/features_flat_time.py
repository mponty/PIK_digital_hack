import pandas as pd
import numpy as np
from common import date2int, inputdir


CROSS_VALIDATION = True

df = pd.read_csv(inputdir+'flat.csv', encoding='cp1251')
df['sale'] = date2int(df['sale'])
df['date_salestart'] = date2int(df['date_salestart'])

if CROSS_VALIDATION:
    future_date = pd.DatetimeIndex(['2017-10-01']).astype(np.int64)[0]
else:
    future_date = pd.DatetimeIndex(['2019-10-01']).astype(np.int64)[0]

# group by 'id_bulk', 'spalen'


def ff(x):
    id_bulk = x['id_bulk'].values[0]
    spalen = x['spalen'].values[0]

    month_num = (x['date_salestart'].values[0] - future_date) / (86400 * 30)

    not_sold = np.sum(x[x['sale'] >= future_date]['square'].values)

    avg_sale_date = np.median(x[x['sale'] < future_date]['sale'].values)
    return pd.Series([id_bulk, spalen, not_sold, avg_sale_date], index=['id_bulk', 'spalen', 'not_sold', 'avg_sale_date'])


f1 = df.groupby(['id_bulk', 'spalen'], as_index=False).apply(ff)
f1 = f1.rename(columns={'id_bulk': 'bulk_id'})

if CROSS_VALIDATION:
    f1.to_csv('features_flat_time_cv.csv', index=False)
else:
    f1.to_csv('features_flat_time.csv', index=False)
