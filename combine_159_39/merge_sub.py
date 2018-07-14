import pandas as pd


test = pd.read_csv('../files-pik_digital_day/test.csv', encoding='cp1251')

# get id of next month
ids_m1 = test[test['date1'] == '2018-03-01']['id'].values
ids_m2 = test[test['date1'] == '2018-04-01']['id'].values
ids_m3 = test[test['date1'] == '2018-05-01']['id'].values

sub_m3 = pd.read_csv('submission_month3.csv')
sub_m2 = pd.read_csv('submission_month2.csv')
sub_m1 = pd.read_csv('submission_month1.csv')

sub_m3 = sub_m3[sub_m3['id'].isin(ids_m3)]
sub_m2 = sub_m2[sub_m2['id'].isin(ids_m2)]
sub_m1 = sub_m1[sub_m1['id'].isin(ids_m1)]

sub_final = pd.concat([sub_m1, sub_m2, sub_m3], ignore_index=True)
sub_final = sub_final.sort_values(by=['id'])

# print(sub_q3.head())
print('final submission ready')

sub_final.to_csv('submission_final.csv', index=False)
