import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool

from features_helper import name_to_col_num


CROSS_VALIDATION = False

train = pd.read_csv('FINAL_TRAIN_month2.csv')
test = pd.read_csv('FINAL_TEST_month2.csv')


# getting cat features indexes
cat_ff = ['date1', 'month', 'Класс объекта', 'Огорожена территория', 'Входные группы', 'Спортивная площадка',
          'Автомойка', 'Кладовые', 'Колясочные', 'Кондиционирование', 'Вентлияция', 'Лифт', 'Система мусоротведения',
          'Видеонаблюдение', 'Подземная парковка', 'Двор без машин', 'most_otdelka', 'most_vid', 'most_plan_size']
cat_ff = name_to_col_num(train.drop(['value', 'bulk_id'], axis=1), cat_ff)

if CROSS_VALIDATION:
    model = CatBoostRegressor(random_state=5, iterations=1000)
    # model = CatBoostRegressor(random_state=1, iterations=1300, learning_rate=0.03, depth=10)

    local_validation_cutoff = pd.DatetimeIndex(['2018-01-01']).astype(np.int64)[0]
    X_train = train[train.date1 < local_validation_cutoff].drop(['value', 'bulk_id'], axis=1)
    y_train = train[train.date1 < local_validation_cutoff]['value']
    X_validation = train[train.date1 >= local_validation_cutoff].drop(['value', 'bulk_id'], axis=1)
    y_validation = train[train.date1 >= local_validation_cutoff]['value']

    f_pool = Pool(X_train, y_train, cat_features=cat_ff)
    model.fit(X_train, y_train,
              cat_features=cat_ff,
              eval_set=(X_validation, y_validation),
              )
    print('best iteration found')

    # feature_importances = model.get_feature_importance(X_train, y_train, cat_features=cat_ff)
    feature_importances = model.get_feature_importance(f_pool)
    feature_names = X_train.columns
    for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
        print('{}: {}'.format(name, score))

else:
    # train = train[train['value'] < 3000]
    X_train = train.drop(['value', 'bulk_id'], axis=1)

    X_test = test.drop(['id', 'bulk_id'], axis=1)
    y_train = train['value']

    preds = []
    for i in range(3):
        # model = CatBoostRegressor(random_state=i, iterations=1300, learning_rate=0.05)
        model = CatBoostRegressor(random_state=i, iterations=800)
        model.fit(X_train, y_train, cat_features=cat_ff)
        preds.append(model.predict(X_test))

    predsub = np.mean(np.array(preds), axis=0)

    sub = pd.DataFrame(columns=['id', 'value'])
    sub['id'] = test['id']


    sub['value'] = np.round(predsub, 4)
    sub.loc[sub['value'] < 0, 'value'] = 0

    sub.to_csv('submission_month2.csv', index=False)

    print(np.mean(predsub))

    # OLD version
    # model = CatBoostRegressor(random_state=333, iterations=1000)
    # model.fit(
    #     train.drop(['value', 'bulk_id'], axis=1), train['value'],
    #     cat_features=cat_ff
    # )
    #
    # prediction = model.predict(test.drop(['id', 'bulk_id'], axis=1))
    #
    # sub = pd.DataFrame(columns=['id', 'value'])
    # sub['id'] = test['id']
    # # sub['value'] = np.round(np.mean(train['value'].values), 4)
    # print(np.mean(train['value'].values))
    # print(np.mean(prediction))
    # sub['value'] = np.round(prediction, 4)
    #
    # sub.loc[sub['value'] < 0, 'value'] = 0
    #
    # sub.to_csv('submission.csv', index=False)
    # END OLD VERSION

    print('SUBMISSION ready!')
