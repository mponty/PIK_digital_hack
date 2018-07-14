import numpy as np
import pandas as pd


flat_df = pd.read_csv('../files-pik_digital_day/flat.csv', encoding='cp1251')

flat_df['date_salestart'] = pd.to_datetime(flat_df['date_salestart'])
flat_df['date_salestart'] = pd.DatetimeIndex(flat_df['date_salestart']).astype(np.int64)
flat_df['flat_startsale'] = pd.to_datetime(flat_df['flat_startsale'])
flat_df['flat_startsale'] = pd.DatetimeIndex(flat_df['flat_startsale']).astype(np.int64)
flat_df['sale'] = pd.to_datetime(flat_df['sale'])
flat_df['sale'] = pd.DatetimeIndex(flat_df['sale']).astype(np.int64)


mm = [
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
    '2018-05-01',
    '2018-06-01',
]

# every_month_stat = flat_df[['bulk_id', 'spalen']].drop_duplicates()
# f_salestart = pd.DataFrame(columns=['bulk_id', 'spalen', 'date1', 'new_still_sale', 'sum_still_sale', 'idle_square'])


# f_salestart = flat_df[['bulk_id', 'spalen', 'date1']].drop_duplicates()
what_we_want = flat_df[['bulk_id', 'spalen']].drop_duplicates()
f_salestart = None
for i, m in enumerate(mm):
    if m == '2018-06-01':
        continue

    new_one = what_we_want.copy()
    new_one['date1'] = pd.DatetimeIndex([m]).astype(np.int64)[0]

    if f_salestart is None:
        f_salestart = new_one.copy()
    else:
        f_salestart = pd.concat([f_salestart, new_one], ignore_index=True)

f_salestart['new_still_sale'] = 0
f_salestart['sum_still_sale'] = 0
f_salestart['sum_flat_still_sale'] = 0
f_salestart['flat_still_sale'] = 0
f_salestart['idle_square'] = 0
f_salestart['shadow_start_square'] = 0

for i, m in enumerate(mm):
    print(m)
    if m == '2018-06-01':
        continue

    cur_m = pd.DatetimeIndex([m]).astype(np.int64)[0]
    next_m = pd.DatetimeIndex([mm[i+1]]).astype(np.int64)[0]

    cur_in_sale = flat_df[flat_df['sale'] >= next_m]
    cur_in_sale = cur_in_sale[cur_in_sale['date_salestart'] < next_m]
    flat_in_sale = flat_df[flat_df['sale'] >= next_m]
    flat_in_sale = flat_in_sale[flat_in_sale['flat_startsale'] < next_m]  # cur_m

    # how many started exactly that month ?
    that_month = cur_in_sale[cur_in_sale['date_salestart'] >= cur_m]
    that_month = that_month.groupby(['bulk_id', 'spalen'], as_index=False).agg({'square': np.sum, 'date_salestart': lambda s: np.mean(s.values - cur_m) / (next_m - cur_m)})
    that_month['square'] = that_month['square'] * that_month['date_salestart']
    that_month = that_month.rename(columns={'square': 'new_still_sale2'})
    that_month = that_month.drop(['date_salestart'], axis=1)
    that_month['date1'] = cur_m
    f_salestart = pd.merge(left=f_salestart, right=that_month, on=['bulk_id', 'spalen', 'date1'], how='left')


    # how many square not still sold ?
    still_not_sold = cur_in_sale.groupby(['bulk_id', 'spalen'], as_index=False).agg({'square': np.sum})
    still_not_sold = still_not_sold.rename(columns={'square': 'sum_still_sale2'})
    still_not_sold['date1'] = cur_m
    f_salestart = pd.merge(left=f_salestart, right=still_not_sold, on=['bulk_id', 'spalen', 'date1'], how='left')

    # how many square not still sold ? NEW DATA
    sum_flat_still_sale = flat_in_sale.groupby(['bulk_id', 'spalen'], as_index=False).agg({'square': np.sum})
    sum_flat_still_sale = sum_flat_still_sale.rename(columns={'square': 'sum_flat_still_sale2'})
    sum_flat_still_sale['date1'] = cur_m
    f_salestart = pd.merge(left=f_salestart, right=sum_flat_still_sale, on=['bulk_id', 'spalen', 'date1'], how='left')

    # the same but number of flats
    # flat_still_sale = cur_in_sale.groupby(['bulk_id', 'spalen'], as_index=False).agg({'floor': np.mean})
    flat_still_sale = cur_in_sale.groupby(['bulk_id', 'spalen'], as_index=False).agg({'floor': lambda s: s[(s > 1) & (s < 8)].values.shape[0]})
    flat_still_sale = flat_still_sale.rename(columns={'floor': 'flat_still_sale2'})
    flat_still_sale['date1'] = cur_m
    f_salestart = pd.merge(left=f_salestart, right=flat_still_sale, on=['bulk_id', 'spalen', 'date1'], how='left')


    # shadow start square
    cur_in_sale2 = flat_df[flat_df['sale'] >= next_m]
    cur_in_sale2 = cur_in_sale2[cur_in_sale2['date_salestart'] < cur_m]
    shadow_start_square = cur_in_sale2.groupby(['bulk_id', 'spalen'], as_index=False).agg({'square': np.sum})
    shadow_start_square = shadow_start_square.rename(columns={'square': 'shadow_start_square2'})
    shadow_start_square['date1'] = cur_m
    f_salestart = pd.merge(left=f_salestart, right=shadow_start_square, on=['bulk_id', 'spalen', 'date1'], how='left')


    # point, that there are no more flats with feature date_salestart ?
    point_all_flats_on_sell = flat_df[flat_df['date_salestart'] >= next_m]
    point_all_flats_on_sell = point_all_flats_on_sell.groupby(['bulk_id', 'spalen'], as_index=False).agg({'square': np.sum})
    point_all_flats_on_sell = point_all_flats_on_sell.rename(columns={'square': 'idle_square2'})
    point_all_flats_on_sell['date1'] = cur_m
    f_salestart = pd.merge(left=f_salestart, right=point_all_flats_on_sell, on=['bulk_id', 'spalen', 'date1'], how='left')

    f_salestart = f_salestart.fillna(0)
    f_salestart['new_still_sale'] = f_salestart['new_still_sale'] + f_salestart['new_still_sale2']
    f_salestart['sum_still_sale'] = f_salestart['sum_still_sale'] + f_salestart['sum_still_sale2']
    f_salestart['sum_flat_still_sale'] = f_salestart['sum_flat_still_sale'] + f_salestart['sum_flat_still_sale2']
    f_salestart['flat_still_sale'] = f_salestart['flat_still_sale'] + f_salestart['flat_still_sale2']
    f_salestart['idle_square'] = f_salestart['idle_square'] + f_salestart['idle_square2']
    f_salestart['shadow_start_square'] = f_salestart['shadow_start_square'] + f_salestart['shadow_start_square2']
    f_salestart = f_salestart.drop(['new_still_sale2', 'sum_still_sale2', 'idle_square2', 'shadow_start_square2', 'flat_still_sale2', 'sum_flat_still_sale2'], axis=1)


f_salestart.fillna(0, inplace=True)
f_salestart = f_salestart.rename(columns={'bulk_id': 'bulk_id'})
f_salestart.to_csv('features_salestart.csv', index=False)

print(__file__)
