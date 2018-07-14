import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from scipy.linalg import norm


train = pd.read_csv('FINAL_TRAIN_month3.csv')
test = pd.read_csv('FINAL_TEST_month3.csv')

# getting cat features indexes
cat_ff_list = ['date1', 'month', 'Класс объекта', 'Огорожена территория', 'Входные группы', 'Спортивная площадка',
          'Автомойка', 'Кладовые', 'Колясочные', 'Кондиционирование', 'Вентлияция', 'Лифт', 'Система мусоротведения',
          'Видеонаблюдение', 'Подземная парковка', 'Двор без машин', 'most_otdelka', 'most_vid', 'most_plan_size']


model = Lasso(0.096)
#model = Ridge(0.01)

# Cross-validation
train = train.dropna(axis=0)
local_validation_cutoff = pd.DatetimeIndex(['2017-10-01']).astype(np.int64)[0]
ix_train = train.date1 < local_validation_cutoff
ix_validation = train.date1 >= local_validation_cutoff

le = LabelEncoder()
for ff in cat_ff_list:
    vals_le = le.fit_transform(train[ff])
    train[ff] = vals_le

X_train = train[ix_train].drop(['value', 'bulk_id'], axis=1)
y_train = train[ix_train]['value']

X_validation = train[ix_validation].drop(['value', 'bulk_id'], axis=1)
y_validation = train[ix_validation]['value']

sc = MinMaxScaler()
sc.fit(train.drop(['value', 'bulk_id'], axis=1))
X_train_sc = sc.transform(X_train)
X_validation_sc = sc.transform(X_validation)

model.fit(X_train_sc, y_train)
y = model.predict(X_validation_sc)
e = norm(y-y_validation)/np.sqrt(len(y_validation))
print('cols = ', X_train.columns)
print('w = ', model.coef_)
print('b = ', model.intercept_)
print('R2 = ', model.score(X_validation_sc, y_validation))
print('e = ', e)



# Prepare submission
le = LabelEncoder()
for ff in cat_ff_list:
    if test[ff].hasnans:
        val_counts = test[ff].value_counts()
        most_frequent_val = val_counts.index[0]
        test[ff] = test[ff].fillna(most_frequent_val)
    vals_le = le.fit_transform(test[ff])
    test[ff] = vals_le

X_test = test.drop(['id', 'bulk_id'], axis=1)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

X_test_sc = sc.transform(X_test)
y = model.predict(X_test_sc)

sub = pd.DataFrame(columns=['id', 'value'])
sub['id'] = test['id']

sub['value'] = np.round(y, 4)
sub.loc[sub['value'] < 0, 'value'] = 0

sub.to_csv('lr_submission_month3!.csv', index=False)
