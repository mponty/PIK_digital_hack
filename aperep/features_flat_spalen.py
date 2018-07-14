import pandas as pd
import numpy as np
from common import date2int, inputdir

df = pd.read_csv(inputdir+'flat.csv', encoding='cp1251')
dfg = df.groupby(['id_bulk', 'spalen'], as_index=False)

square_median = dfg.agg({'square': np.median}).rename(columns={'square': 'square_median'})
square_sum = dfg.agg({'square': np.sum}).rename(columns={'square': 'square_sum'})

num_flats = dfg.agg({'square': len}).rename(columns={'square': 'num_flats'})

square_min = dfg.agg({'square': np.min}).rename(columns={'square': 'square_min'})
floor_median = dfg.agg({'floor': np.median}).rename(columns={'floor': 'floor_median'})

number_floor_1 = dfg.agg({'floor': lambda s: s[s <= 1].values.shape[0]}).rename(columns={'floor': 'number_floor_1'})
number_floor_16 = dfg.agg({'floor': lambda s: s[s >= 16].values.shape[0]}).rename(columns={'floor': 'number_floor_16'})

under_40 = dfg.agg({'square': lambda s: len(s[s < 40]) / len(s) }).rename(columns={'square': 'under_40'})
under_50 = dfg.agg({'square': lambda s: len(s[s < 50]) / len(s) }).rename(columns={'square': 'under_50'})
under_60 = dfg.agg({'square': lambda s: len(s[(s < 60) & (s >= 40)]) / len(s) }).rename(columns={'square': 'under_60'})
under_80 = dfg.agg({'square': lambda s: len(s[(s < 80) & (s >= 60)]) / len(s) }).rename(columns={'square': 'under_80'})
under_100 = dfg.agg({'square': lambda s: len(s[(s < 100) & (s >= 80)]) / len(s) }).rename(columns={'square': 'under_100'})
over_100 = dfg.agg({'square': lambda s: len(s[s >= 100]) / len(s) }).rename(columns={'square': 'over_100'})


most_otdelka = dfg.agg({'otdelka': lambda s: s.fillna('--').value_counts().keys()[0] }).rename(columns={'otdelka': 'most_otdelka'})
most_vid = dfg.agg({'vid': lambda s: s.fillna('--').value_counts().keys()[0]}).rename(columns={'vid': 'most_vid'})
most_plan_size = dfg.agg({'plan_size': lambda s: s.fillna('--').value_counts().keys()[0]}).rename(columns={'plan_size': 'most_plan_size'})


df['date_salestart'] = date2int(df['date_salestart'])
df['date_settle'] = date2int(df['date_settle'])
df['stroy_time'] = (df['date_settle'] - df['date_salestart']) / 1000000000
dfg = df.groupby(['id_bulk', 'spalen'], as_index=False)
stroy_time = dfg.agg({'stroy_time': lambda s: s.values[0]})
date_salestart = dfg.agg({'date_salestart': lambda s: s.values[0]})
date_settle = dfg.agg({'date_settle': lambda s: s.values[0]})


final = square_median
for r in [square_min, floor_median, square_sum, num_flats, number_floor_1, number_floor_16, most_otdelka, most_vid, most_plan_size, stroy_time, date_salestart, date_settle]:
  final = pd.merge(left=final, right=r, on=['id_bulk', 'spalen'], how='left')

final = final.rename(columns={'id_bulk': 'bulk_id'})

final.to_csv('features_flat_spalen.csv', index=False)

print(__file__)
