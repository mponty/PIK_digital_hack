import pandas as pd
import numpy as np


df = pd.read_csv('../files-pik_digital_day/flat.csv', encoding='cp1251')


square_median = df.groupby(['bulk_id', 'spalen'], as_index=False).agg({'square': np.median})
square_median = square_median.rename(columns={'square': 'square_median'})

square_sum = df.groupby(['bulk_id', 'spalen'], as_index=False).agg({'square': np.sum})
square_sum = square_sum.rename(columns={'square': 'square_sum'})

num_flats = df.groupby(['bulk_id', 'spalen'], as_index=False).agg({'square': len})
num_flats = num_flats.rename(columns={'square': 'num_flats'})

square_min = df.groupby(['bulk_id', 'spalen'], as_index=False).agg({'square': np.min})
square_min = square_min.rename(columns={'square': 'square_min'})

floor_median = df.groupby(['bulk_id', 'spalen'], as_index=False).agg({'floor': np.median})
floor_median = floor_median.rename(columns={'floor': 'floor_median'})

# floor_max = df.groupby(['bulk_id'], as_index=False).agg({'floor': np.max})
# floor_max = floor_max.rename(columns={'floor': 'floor_max'})

# floor_min = df.groupby(['bulk_id'], as_index=False).agg({'floor': np.max})
# floor_min = floor_min.rename(columns={'floor': 'floor_max'})

number_floor_1 = df.groupby(['bulk_id', 'spalen'], as_index=False).agg({'floor': lambda s: s[s <= 1].values.shape[0]})
number_floor_1 = number_floor_1.rename(columns={'floor': 'number_floor_1'})

number_floor_16 = df.groupby(['bulk_id', 'spalen'], as_index=False).agg({'floor': lambda s: s[s >= 16].values.shape[0]})
number_floor_16 = number_floor_16.rename(columns={'floor': 'number_floor_16'})


under_40 = df.groupby(['bulk_id', 'spalen'], as_index=False).agg({'square': lambda s: len(s[s < 40]) / len(s) })
under_40 = under_40.rename(columns={'square': 'under_40'})

under_50 = df.groupby(['bulk_id', 'spalen'], as_index=False).agg({'square': lambda s: len(s[s < 50]) / len(s) })
under_50 = under_50.rename(columns={'square': 'under_50'})

under_60 = df.groupby(['bulk_id', 'spalen'], as_index=False).agg({'square': lambda s: len(s[(s < 60) & (s >= 40)]) / len(s) })
under_60 = under_60.rename(columns={'square': 'under_60'})

# under_80 = df.groupby(['bulk_id'], as_index=False).agg({'square': lambda s: len(s[(s < 80) & (s >= 60)]) / len(s) })
under_80 = df.groupby(['bulk_id', 'spalen'], as_index=False).agg({'square': lambda s: len(s[(s < 80) & (s >= 60)]) / len(s) })
under_80 = under_80.rename(columns={'square': 'under_80'})

under_100 = df.groupby(['bulk_id', 'spalen'], as_index=False).agg({'square': lambda s: len(s[(s < 100) & (s >= 80)]) / len(s) })
under_100 = under_100.rename(columns={'square': 'under_100'})

over_100 = df.groupby(['bulk_id', 'spalen'], as_index=False).agg({'square': lambda s: len(s[s >= 100]) / len(s) })
over_100 = over_100.rename(columns={'square': 'over_100'})


most_otdelka = df.groupby(['bulk_id', 'spalen'], as_index=False).agg({'otdelka': lambda s: s.fillna('--').value_counts().keys()[0] })
most_otdelka = most_otdelka.rename(columns={'otdelka': 'most_otdelka'})

most_vid = df.groupby(['bulk_id', 'spalen'], as_index=False).agg({'vid': lambda s: s.fillna('--').value_counts().keys()[0]})
most_vid = most_vid.rename(columns={'vid': 'most_vid'})

most_plan_size = df.groupby(['bulk_id', 'spalen'], as_index=False).agg({'plan_size': lambda s: s.fillna('--').value_counts().keys()[0]})
most_plan_size = most_plan_size.rename(columns={'plan_size': 'most_plan_size'})

# id_gk = df[['bulk_id', 'id_gk']].drop_duplicates()


df['date_salestart'] = pd.to_datetime(df['date_salestart'])
df['date_salestart'] = pd.DatetimeIndex(df['date_salestart']).astype(np.int64)
df['date_settle'] = pd.to_datetime(df['date_settle'])
df['date_settle'] = pd.DatetimeIndex(df['date_settle']).astype(np.int64)
df['stroy_time'] = (df['date_settle'] - df['date_salestart']) / 1000000000
stroy_time = df.groupby(['bulk_id', 'spalen'], as_index=False).agg({'stroy_time': lambda s: s.values[0]})
date_salestart = df.groupby(['bulk_id', 'spalen'], as_index=False).agg({'date_salestart': lambda s: s.values[0]})
date_settle = df.groupby(['bulk_id', 'spalen'], as_index=False).agg({'date_settle': lambda s: s.values[0]})


final = pd.merge(left=square_median, right=square_min, on=['bulk_id', 'spalen'], how='left')
final = pd.merge(left=final, right=floor_median, on=['bulk_id', 'spalen'], how='left')
final = pd.merge(left=final, right=square_sum, on=['bulk_id', 'spalen'], how='left')
final = pd.merge(left=final, right=num_flats, on=['bulk_id', 'spalen'], how='left')
# final = pd.merge(left=final, right=floor_max, on='bulk_id', how='left')
# final = pd.merge(left=final, right=floor_min, on='bulk_id', how='left')
final = pd.merge(left=final, right=number_floor_1, on=['bulk_id', 'spalen'], how='left')
final = pd.merge(left=final, right=number_floor_16, on=['bulk_id', 'spalen'], how='left')
# final = pd.merge(left=final, right=under_40, on=['bulk_id', 'spalen'], how='left')
# final = pd.merge(left=final, right=under_50, on='bulk_id', how='left')
# final = pd.merge(left=final, right=under_60, on=['bulk_id', 'spalen'], how='left')
# final = pd.merge(left=final, right=under_80, on=['bulk_id', 'spalen'], how='left')
# final = pd.merge(left=final, right=under_100, on=['bulk_id', 'spalen'], how='left')
# final = pd.merge(left=final, right=over_100, on=['bulk_id', 'spalen'], how='left')

final = pd.merge(left=final, right=most_otdelka, on=['bulk_id', 'spalen'], how='left')
final = pd.merge(left=final, right=most_vid, on=['bulk_id', 'spalen'], how='left')
final = pd.merge(left=final, right=most_plan_size, on=['bulk_id', 'spalen'], how='left')


final = pd.merge(left=final, right=stroy_time, on=['bulk_id', 'spalen'], how='left')
final = pd.merge(left=final, right=date_salestart, on=['bulk_id', 'spalen'], how='left')
final = pd.merge(left=final, right=date_settle, on=['bulk_id', 'spalen'], how='left')

# final = pd.merge(left=final, right=id_gk, on=['bulk_id'], how='left')


final = final.rename(columns={'bulk_id': 'bulk_id'})

final.to_csv('features_flat_spalen.csv', index=False)

print(__file__)
