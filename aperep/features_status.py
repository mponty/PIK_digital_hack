import numpy as np
import pandas as pd
from common import date2int, inputdir


flat_df = pd.read_csv(inputdir+'flat.csv', encoding='cp1251')
stat_df = pd.read_csv(inputdir+'status.csv', encoding='cp1251')

q_periods = {
    ('2015-08-01', '2015-08-01'): ['2015-08-01', '2015-09-01', '2015-10-01'],
    ('2015-08-01', '2015-11-01'): ['2015-11-01', '2015-12-01', '2016-01-01'],
    ('2015-11-01', '2016-02-01'): ['2016-02-01', '2016-03-01', '2016-04-01'],
    ('2016-02-01', '2016-05-01'): ['2016-05-01', '2016-06-01', '2016-07-01'],
    ('2016-05-01', '2016-08-01'): ['2016-08-01', '2016-09-01', '2016-10-01'],
    ('2016-08-01', '2016-11-01'): ['2016-11-01', '2016-12-01', '2017-01-01'],
    ('2016-11-01', '2017-02-01'): ['2017-02-01', '2017-03-01', '2017-04-01'],
    ('2017-02-01', '2017-05-01'): ['2017-05-01', '2017-06-01', '2017-07-01'],
    ('2017-05-01', '2017-08-01'): ['2017-08-01', '2017-09-01', '2017-10-01'],
    ('2017-08-01', '2017-11-01'): ['2017-11-01', '2017-12-01', '2018-01-01'],
    ('2017-11-01', '2018-02-01'): ['2018-02-01', '2018-03-01', '2018-04-01']
}

stat_df['dateto'] = date2int(stat_df['dateto'])
stat_df['datefrom'] = date2int(stat_df['datefrom'])
flat_df['sale'] = date2int(flat_df['sale'])

# what we want:
# bulk_id, spalen, date1, ratio of status 03 for bulk_id
flat_df = pd.merge(left=flat_df, right=stat_df, on='id_flatwork', how='left')

what_we_want = flat_df[['id_bulk', 'spalen']].drop_duplicates()
ff = None

for d1, d123 in q_periods.items():
    for dd in d123:
        new_one = what_we_want.copy()
        new_one['date1'] = pd.DatetimeIndex([dd]).astype(np.int64)[0]
        if ff is None:
            ff = new_one.copy()
        else:
            ff = pd.concat([ff, new_one], ignore_index=True)

ff['ratio_zar'] = 0
ff['ratio_vre'] = 0
ff['ratio_vre_16'] = 0
ff['ratio_nre'] = 0
ff['ratio_vsre'] = 0
ff['sold_in_quarter'] = 0

for d1, d123 in q_periods.items():
    start_p = pd.DatetimeIndex([d1[0]]).astype(np.int64)[0]
    end_p = pd.DatetimeIndex([d1[1]]).astype(np.int64)[0]

    zar = flat_df[flat_df['stat_name'] == 'Зарезервирован под клиента']
    zar = zar[(zar['datefrom'] >= start_p) & (zar['datefrom'] < end_p)]
    zar = zar.groupby(['id_bulk', 'spalen'], as_index=False).agg({'id_flatwork': lambda s: s.drop_duplicates().values.shape[0]})
    zar = zar.rename(columns={'id_flatwork': 'ratio_zar2'})

    v_re = flat_df[flat_df['stat_name'] == 'В реализации']
    v_re = v_re[(v_re['datefrom'] >= start_p) & (v_re['datefrom'] < end_p)]
    v_re = v_re.groupby(['id_bulk', 'spalen'], as_index=False).agg({'id_flatwork': lambda s: s.drop_duplicates().values.shape[0]})
    v_re = v_re.rename(columns={'id_flatwork': 'ratio_vre2'})

    v_re_16 = flat_df[(flat_df['stat_name'] == 'В реализации') & (flat_df['floor'] >= 16)]
    v_re_16 = v_re_16[(v_re_16['datefrom'] >= start_p) & (v_re_16['datefrom'] < end_p)]
    v_re_16 = v_re_16.groupby(['id_bulk', 'spalen'], as_index=False).agg({'id_flatwork': lambda s: s.drop_duplicates().values.shape[0]})
    v_re_16 = v_re_16.rename(columns={'id_flatwork': 'ratio_vre_162'})

    ne_re = flat_df[flat_df['stat_name'] == 'Не реализуется']
    ne_re = ne_re[(ne_re['datefrom'] >= start_p) & (ne_re['datefrom'] < end_p)]
    ne_re = ne_re.groupby(['id_bulk', 'spalen'], as_index=False).agg({'id_flatwork': lambda s: s.drop_duplicates().values.shape[0]})
    ne_re = ne_re.rename(columns={'id_flatwork': 'ratio_nre2'})


    # does not add accuracy :(
    vs_re = flat_df[flat_df['stat_name'] == 'Платное бронирование']
    vs_re = vs_re[(vs_re['datefrom'] >= start_p) & (vs_re['datefrom'] < end_p)]
    vs_re = vs_re.groupby(['id_bulk', 'spalen'], as_index=False).agg({'id_flatwork': lambda s: s.drop_duplicates().values.shape[0]})
    vs_re = vs_re.rename(columns={'id_flatwork': 'ratio_vsre2'})

    # sold
    sold_in_quarter = flat_df[(flat_df['sale'] >= start_p) & (flat_df['sale'] < end_p)]
    sold_in_quarter = sold_in_quarter.groupby(['id_bulk', 'spalen'], as_index=False).agg({'square': np.sum})
    sold_in_quarter = sold_in_quarter.rename(columns={'square': 'sold_in_quarter2'})

    for dd in d123:
        zar['date1'] = pd.DatetimeIndex([dd]).astype(np.int64)[0]
        ff = pd.merge(left=ff, right=zar, on=['id_bulk', 'spalen', 'date1'], how='left')
        ff = ff.fillna(0)
        ff['ratio_zar'] = ff['ratio_zar'] + ff['ratio_zar2']
        ff = ff.drop(['ratio_zar2'], axis=1)

        v_re['date1'] = pd.DatetimeIndex([dd]).astype(np.int64)[0]
        ff = pd.merge(left=ff, right=v_re, on=['id_bulk', 'spalen', 'date1'], how='left')
        ff = ff.fillna(0)
        ff['ratio_vre'] = ff['ratio_vre'] + ff['ratio_vre2']
        ff = ff.drop(['ratio_vre2'], axis=1)

        v_re_16['date1'] = pd.DatetimeIndex([dd]).astype(np.int64)[0]
        ff = pd.merge(left=ff, right=v_re_16, on=['id_bulk', 'spalen', 'date1'], how='left')
        ff = ff.fillna(0)
        ff['ratio_vre_16'] = ff['ratio_vre_16'] + ff['ratio_vre_162']
        ff = ff.drop(['ratio_vre_162'], axis=1)

        ne_re['date1'] = pd.DatetimeIndex([dd]).astype(np.int64)[0]
        ff = pd.merge(left=ff, right=ne_re, on=['id_bulk', 'spalen', 'date1'], how='left')
        ff = ff.fillna(0)
        ff['ratio_nre'] = ff['ratio_nre'] + ff['ratio_nre2']
        ff = ff.drop(['ratio_nre2'], axis=1)

        vs_re['date1'] = pd.DatetimeIndex([dd]).astype(np.int64)[0]
        ff = pd.merge(left=ff, right=vs_re, on=['id_bulk', 'spalen', 'date1'], how='left')
        ff = ff.fillna(0)
        ff['ratio_vsre'] = ff['ratio_vsre'] + ff['ratio_vsre2']
        ff = ff.drop(['ratio_vsre2'], axis=1)

        # ++++
        sold_in_quarter['date1'] = pd.DatetimeIndex([dd]).astype(np.int64)[0]
        ff = pd.merge(left=ff, right=sold_in_quarter, on=['id_bulk', 'spalen', 'date1'], how='left')
        ff = ff.fillna(0)
        ff['sold_in_quarter'] = ff['sold_in_quarter'] + ff['sold_in_quarter2']
        ff = ff.drop(['sold_in_quarter2'], axis=1)



ff = ff.rename(columns={'id_bulk': 'bulk_id'})

ff.to_csv('features_status.csv', index=False)

print(__file__)
