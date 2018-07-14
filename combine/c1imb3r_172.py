import sys
import gc
sys.path.append('/usr/local/lib/python3.6/site-packages')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, mean_squared_error, make_scorer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


import lightgbm as lgb
#from hyperopt import hp, tpe
#from hyperopt.fmin import fmin

gc.collect()


def mem_economy(dataset):
    features = np.array(dataset.dtypes[dataset.dtypes=='float64'].index)
    for f in features:
        dataset[f] = dataset[f].astype('float32')
    features = np.array(dataset.dtypes[dataset.dtypes=='int64'].index)
    for f in features:
        dataset[f] = dataset[f].astype('int32')
    
    return dataset

def my_drop_levels(dataset, sep = '_', brief = ''):
    if dataset.columns.nlevels>1:
        new_columns = np.array([], dtype = 'str')
        for col_i in range(dataset.shape[1]):
            col_name = brief
            for level in range(dataset.columns.nlevels):
                tmp_col_name = dataset.columns.levels[level][dataset.columns.labels[level][col_i]]#.astype('str')
                tmp_col_name = str(tmp_col_name) 
                if (level>0) & (tmp_col_name!=''):
                    col_name = col_name+'_'
                col_name = col_name+tmp_col_name
            new_columns = np.append(new_columns,col_name)
            #print(col_name)
        for level in range(dataset.columns.nlevels-1):
            dataset.columns.droplevel(0)
        dataset.columns = new_columns      
    return dataset

def prepare_full():
    train = pd.read_csv('../files-pik_digital_day/train.csv', encoding='cp1251')
    test = pd.read_csv('../files-pik_digital_day/test.csv', encoding='cp1251')
    
    train['is_train'] = 1
    test['is_train'] = 0
    
    full = pd.concat([train,test])
    
    del train, test
    gc.collect()
    
    full = full.sort_values('month_cnt')
    
    le = LabelEncoder()
    full['bulk_id_int'] = le.fit_transform(full['bulk_id'])
    full['date1'] = pd.to_datetime(full['date1'], format='%Y-%m-%d')
    GLOBAL_MINDATE = full['date1'].min()
    full['Date_int'] = ((full['date1'] - GLOBAL_MINDATE)/np.timedelta64(1, 'D')).astype('int32')

    full['Автомойка'] = (full['Автомойка']=='да').astype('int')
    full['Входные группы'] = (full['Входные группы']=='да').astype('int')
    full['Двор без машин'] = (full['Двор без машин']=='да').astype('int')
    full['Класс объекта'] = full['Класс объекта'].map({'эконом':1, 'комфорт':3, 'стандарт':2})
    full['Кладовые'] = (full['Кладовые']=='да').astype('int')
    full['Колясочные'] = (full['Колясочные']=='да').astype('int')
    full['Огорожена территория'] = (full['Огорожена территория']=='да').astype('int')
    full['Подземная парковка'] = (full['Подземная парковка']=='да').astype('int')
    full['Система мусоротведения'] = le.fit_transform(full['Система мусоротведения'])
    full['Спортивная площадка'] = (full['Спортивная площадка']=='да').astype('int')
    
    #введем уникальные id
    full['bulk_spalen_id'] = full['bulk_id_int'].astype('str')+'_'+full['spalen'].astype('str')
    full['bulk_spalen_id'] = le.fit_transform(full['bulk_spalen_id'])
     
    # подсчитаем псевдо start_square (без учета возвращенных)
    full['calc_start_square'] = full.groupby(['bulk_spalen_id'])['start_square'].shift(1) - full.groupby(['bulk_spalen_id'])['value'].shift(1)
    full['calc_last_value'] = full.groupby(['bulk_spalen_id'])['value'].shift(1)
    
    full['date2'] = full.date1+ pd.offsets.MonthEnd(1)
    
    full['price_by_square'] = full['price']/full['mean_sq']
    
    
    full = full.reset_index(drop = True)
    
    return full
    

def prepare_flat():
    le = LabelEncoder()
    flat = pd.read_csv('../files-pik_digital_day/flat.csv', encoding='cp1251')
    flat = flat.rename(columns = {'id_bulk':'bulk_id'})
    flat['id_flatwork_int'] = np.array(flat.index).astype('int')
    
    dict_bulk_spalen = full.loc[:, ('bulk_id','bulk_id_int','spalen','bulk_spalen_id')] \
                       .drop_duplicates() 
    dict_flat = flat[['id_flatwork_int','id_flatwork','bulk_id','spalen']].copy()

    dict_flat = dict_flat.merge(dict_bulk_spalen, how = 'left')
    
    
    flat['Автомойка'] = (flat['Автомойка']=='да').astype('int')
    flat['Входные группы'] = (flat['Входные группы']=='да').astype('int')
    flat['Двор без машин'] = (flat['Двор без машин']=='да').astype('int')
    

    flat['Класс объекта'] = flat['Класс объекта'].fillna('эконом').map({'эконом':1, 'комфорт':3, 'стандарт':2}).astype('int')
    flat['Кладовые'] = (flat['Кладовые']=='да').astype('int')
    flat['Колясочные'] = (flat['Колясочные']=='да').astype('int')
    flat['Огорожена территория'] = (flat['Огорожена территория']=='да').astype('int')
    flat['Подземная парковка'] = (flat['Подземная парковка']=='да').astype('int')
    flat.drop('Система мусоротведения', axis = 1, inplace = True)
    #flat['Система мусоротведения'] = le.fit_transform(flat['Система мусоротведения'])
    flat['Спортивная площадка'] = (flat['Спортивная площадка']=='да').astype('int')
    flat['otdelka'] = le.fit_transform(flat['otdelka'].fillna('nan'))
    flat['vid'] = flat['vid'].map({'эконом':1, 'средний':2, 'хороший':3}).fillna(0).astype('int')
    flat['plan_size'] = flat['plan_size'].fillna('-1').map({'S':1, 'M':2, 'L':3, '-1':0}).astype('int')
    flat['plan0'] = le.fit_transform(flat['plan0'].fillna('nan'))
    

    flat['date_flat_startsale'] = pd.to_datetime(flat['flat_startsale'].fillna('2018-03-01'), format='%Y-%m-%d')
    flat['date_settle'] = pd.to_datetime(flat['date_settle'], format='%Y-%m-%d')
    flat['date_salestart'] = pd.to_datetime(flat['date_salestart'].fillna('2018-03-01'), format='%Y-%m-%d')
    
    flat['dt_flat_salestart_delay'] = ((flat['date_flat_startsale'] - flat['date_salestart'])/np.timedelta64(1, 'D')).astype('int32') 
    
    
    flat['sale'] = (pd.to_datetime(flat['sale'], format = '%Y-%m-%d %H:%M:%S') +  np.timedelta64(1,'D') 
                   ).dt.date


    flat.loc[~flat['date_settle'].isna(),'dt_settle_salestart'] = ((flat.loc[~flat['date_settle'].isna(),'date_settle'] - flat.loc[~flat['date_settle'].isna(),'date_salestart'])/np.timedelta64(1, 'D')).astype('int32')

    #заполним медианным значением
    dt_settle_salestart_median = int(flat.loc[~flat['date_settle'].isna(),'dt_settle_salestart'].median())
    flat.loc[flat['date_settle'].isna(),'dt_settle_salestart'] = dt_settle_salestart_median  
    
    flat['dt_settle_salestart'] = flat['dt_settle_salestart'].astype('int32')
    
    flat.loc[flat['date_settle'].isna(),'date_settle'] = flat.loc[flat['date_settle'].isna(),'date_salestart'] + np.timedelta64(dt_settle_salestart_median, 'D')
    
    
    flat = flat.merge(dict_flat[['id_flatwork_int','bulk_spalen_id','bulk_id_int']], how = 'left', on = 'id_flatwork_int')
     
    return flat, dict_bulk_spalen, dict_flat
    
def prepare_status():
    


    status = pd.read_csv('../files-pik_digital_day/status.csv', encoding='cp1251')
    status = status.merge(dict_flat, how = 'inner')

    #удалим статусы-однодневки
    status = status[status['datefrom']!=status['dateto']]

    dict_stat = status[['stat','stat_name']].drop_duplicates()
    dict_stat['can_be_sold'] = (~dict_stat.stat_name.isin(['Реализован','Статус после покупки'])).astype('int16')
    dict_stat['sold'] = (dict_stat.stat_name.isin(['Реализован','Статус после покупки'])).astype('int16')
    dict_stat['realize'] = (dict_stat.stat_name.isin(['Реализован'])).astype('int16')
    dict_stat['stat_new'] = dict_stat['stat_name'].map({'Не реализуется':0,
                                                        'В реализации (не на сайте)':1,
                                                        'В реализации':2,
                                                        'Онлайн бронирование':3,
                                                        'Зарезервирован под клиента':3,
                                                        'Платное бронирование':4,
                                                        'Реализован':5,
                                                        'Статус после покупки':6
                                                        }).astype('int16')

    status = status.merge(dict_stat[['stat','stat_new']], how = 'inner')


    status['datefrom'] = (pd.to_datetime(status['datefrom'], format = '%Y-%m-%d %H:%M:%S') + np.timedelta64(1,'D')).dt.date.astype('str')
    status['dateto'] = (pd.to_datetime(status['dateto'], format = '%Y-%m-%d %H:%M:%S') + np.timedelta64(1,'D')).dt.date.astype('str')


    status['datefrom_dt'] = pd.to_datetime(status['datefrom'], format='%Y-%m-%d')
    status['dateto_dt'] = pd.to_datetime(status['dateto'], format='%Y-%m-%d')


    status = status.sort_values(['datefrom','dateto'])
    status['last_stat_new'] = status.groupby(['id_flatwork_int'])['stat_new'].shift(1)#.fillna('stat_new')
    status.loc[status['last_stat_new'].isna(),'last_stat_new'] = -1



    status['delay'] = (status['dateto_dt']-status['datefrom_dt'])/np.timedelta64(1,'D')


    gc.collect() 
    
    return status, dict_stat
    
def prepare_price():
    
    price = pd.read_csv('../files-pik_digital_day/price.csv', encoding='utf-8')
    price = price.merge(dict_flat, how = 'inner')

    #удалим пустые цены и цены однодневки
    price = price[(price.pricem2>1) & (price['datefrom']!=price['dateto'])].sort_values(['datefrom','dateto'])

    price['datefrom'] = (pd.to_datetime(price['datefrom'], format = '%Y-%m-%d %H:%M:%S') + np.timedelta64(1,'D')).dt.date.astype('str')
    price['dateto'] = (pd.to_datetime(price['dateto'], format = '%Y-%m-%d %H:%M:%S') + np.timedelta64(1,'D')).dt.date.astype('str')


    price['last_pricem2'] = price.groupby(['id_flatwork_int'])['pricem2'].shift(1).fillna(0)
    price['diff_pricem2'] = price['pricem2'] - price['last_pricem2']
    price['was_decrease'] = (price['diff_pricem2'] < 0).astype('int32')



    price['datefrom_dt'] = pd.to_datetime(price['datefrom'], format='%Y-%m-%d')
    price['dateto_dt'] = pd.to_datetime(price['dateto'], format='%Y-%m-%d')

    price = price.merge(flat.loc[:,('sale','id_flatwork_int')], how = 'left')
    price['sale'] = pd.to_datetime(price['sale'], format='%Y-%m-%d')
    price['is_saled_price'] = ((price['sale']>=price['datefrom_dt']) & (price['sale']<price['dateto_dt'])).astype('int')

    price['delay'] = (price['dateto_dt']-price['datefrom_dt'])/np.timedelta64(1,'D')

    gc.collect()
    
    return price   

def prepare_flat_train(test_days_period):
    

    fixed_dates = np.sort(full.date1.dt.strftime('%Y-%m-%d').unique())
    fixed_dates_last = np.sort(full.date2.dt.strftime('%Y-%m-%d').unique())


    for i in range(len(fixed_dates)):
        gc.collect()


        fixed_date = fixed_dates[i] 
        fixed_date_last = fixed_dates_last[i]


        #найдем квартиры, доступные к продаже на эту дату
        status_on_date = status[(status.datefrom<=fixed_date) & (status.dateto>fixed_date)].copy()


        #срок жизни статуса
        status_on_date['datefrom'] = pd.to_datetime(status_on_date['datefrom'], format='%Y-%m-%d')
        status_on_date['datenow'] = pd.to_datetime(fixed_date, format='%Y-%m-%d')
        #status_on_date['dateto'] = pd.to_datetime(status_on_date['dateto'], format='%Y-%m-%d %H:%M:%S')

        status_on_date['status_days'] = ((status_on_date['datenow'] - 
                                          status_on_date['datefrom'])/np.timedelta64(1, 'D')).astype('int32')

        #подсчитаем все статусы, которые были у этой квартиры к указанной дате
        statuses_to_date = status[(status.datefrom<=fixed_date)] \
                        .groupby(['id_flatwork_int','stat_new']) \
                        .size() \
                        .reset_index(name = 'cnt_stat_new')
        statuses_to_date = statuses_to_date.pivot(index = 'id_flatwork_int', columns='stat_new').fillna(0)
        statuses_to_date = my_drop_levels(statuses_to_date, sep = '_') 
        statuses_to_date = statuses_to_date.reset_index()

        #удалим задвоения
        tmp = status_on_date.groupby('id_flatwork_int').size().reset_index(name = 'cnt')
        flats_to_delete =  tmp.loc[tmp['cnt']>1,'id_flatwork_int']
        status_on_date = status_on_date[~status_on_date.id_flatwork_int.isin(flats_to_delete)] 

        #квартиры, которые могут быть проданы в этом периоде
        stats_can_be_sold = dict_stat[dict_stat['can_be_sold']==1].stat
        flats_can_be_sold = np.array(status_on_date[status_on_date.stat.isin(stats_can_be_sold)] \
                                     .id_flatwork_int)

        #  
        stats_sold = dict_stat[dict_stat['sold']==1].stat
        flats_sold = status_on_date[status_on_date.stat.isin(stats_sold)].id_flatwork_int
        
        
        flats_returned = np.array(flat[(flat.sale.astype('str')>fixed_date) & 
                              flat.id_flatwork_int.isin(flats_sold)].id_flatwork_int)
        
        flats_can_be_sold = np.append(flats_can_be_sold, flats_returned)
        
        #формируем простейшую поквартирную обучающую выборку
        #Если реальная дата продажи > чем начало периода, то даже при статусе реализован, она может быть продана
        tmp_flat_train = flat[(~flat.bulk_spalen_id.isna()) & 
                              (flat.date_salestart <= fixed_date_last) & #ВОТ ТУТ ПОМЕНЯТЬ!!!
                              #(flat.flat_salestart <= fixed_date_last) 
                              (flat.id_flatwork_int.isin(flats_can_be_sold))].copy()

        tmp_flat_train['date1']=fixed_date 
        tmp_flat_train['month_cnt']=i
        tmp_flat_train['dt_to_settle'] = ((pd.to_datetime(tmp_flat_train['date1'], 
                                                          format='%Y-%m-%d')  - 
                                           tmp_flat_train['date_settle'])/np.timedelta64(1, 'D')).astype('int32')
        tmp_flat_train['dt_to_salestart'] = ((pd.to_datetime(tmp_flat_train['date1'], format='%Y-%m-%d') - 
                                              tmp_flat_train['date_salestart'])/np.timedelta64(1, 'D')).astype('int32')

        tmp_flat_train['dt_to_sale'] = ((pd.to_datetime(tmp_flat_train['sale'], 
                                                        format='%Y-%m-%d') - 
                                         pd.to_datetime(tmp_flat_train['date1'], 
                                                        format='%Y-%m-%d'))/np.timedelta64(1, 'D')).astype('int32')  

        tmp_flat_train = tmp_flat_train.merge(status_on_date[['id_flatwork_int','stat_new',
                                                              'last_stat_new','status_days']], 
                                              how = 'inner', 
                                              on = 'id_flatwork_int')
        tmp_flat_train = tmp_flat_train.merge(statuses_to_date, how = 'left', on = 'id_flatwork_int')

        #цена на дату
        price_on_date = price[(price.datefrom<=fixed_date) & (price.dateto>fixed_date)].copy()
        price_on_date['datefrom'] = pd.to_datetime(price_on_date['datefrom'], format='%Y-%m-%d')
        price_on_date['datenow'] = pd.to_datetime(fixed_date, format='%Y-%m-%d')
        #status_on_date['dateto'] = pd.to_datetime(status_on_date['dateto'], format='%Y-%m-%d %H:%M:%S')

        price_on_date['price_days'] = ((price_on_date['datenow'] - 
                                        price_on_date['datefrom'])/np.timedelta64(1, 'D')).astype('int32')


        tmp_flat_train = tmp_flat_train.merge(price_on_date[['id_flatwork_int','pricem2',
                                                             'last_pricem2','diff_pricem2',
                                                             'price_days','was_decrease']], 
                                              how = 'inner', 
                                              on = 'id_flatwork_int')


        #исторические движения по цене
        prices_to_date = price[(price.datefrom<=fixed_date) & (price.pricem2>1)] \
                        .groupby(['id_flatwork_int']) \
                        .agg({'pricem2':('min','max','mean','median','std'),
                              'was_decrease':('sum','mean','std')})

        prices_to_date = my_drop_levels(prices_to_date, sep = '_') 
        prices_to_date = prices_to_date.reset_index()
        tmp_flat_train = tmp_flat_train.merge(prices_to_date, how = 'left', on = 'id_flatwork_int')


        if i==0:
            flat_train = tmp_flat_train.fillna(0)
        else:
            flat_train = flat_train.append(tmp_flat_train.fillna(0))

       

    flat_train = flat_train.fillna(0) 


    flat_train['realized_1'] = ((flat_train.dt_to_sale>=0) & 
                                (flat_train.dt_to_sale<test_days_period[0])).astype('int')
    flat_train['realized_2'] = ((flat_train.dt_to_sale>=test_days_period[0]) & 
                                (flat_train.dt_to_sale<test_days_period[0]+test_days_period[1])).astype('int')           
    flat_train['realized_3'] = ((flat_train.dt_to_sale>=test_days_period[0]+test_days_period[1]) & 
                                (flat_train.dt_to_sale<=test_days_period[0]+test_days_period[1]+test_days_period[2])).astype('int') 

    flat_train['value_1'] = flat_train['square']*flat_train['realized_1']
    flat_train['value_2'] = flat_train['square']*flat_train['realized_2']
    flat_train['value_3'] = flat_train['square']*flat_train['realized_3']
    
    
    #может иметь психологический эффект
    flat_train['price'] = flat_train['pricem2']*flat_train['square']

    tmp = flat_train[(flat_train['pricem2']>1) & (flat_train['stat_new']>0) & (flat_train['stat_new']<5)] \
                    .groupby(['month_cnt','bulk_spalen_id']) \
                    .agg({'price':('min','max','std','count','median'),
                          'pricem2':('min','max','std','median')})

    tmp = my_drop_levels(tmp, sep = '_', brief = 'bulk_spalen_').reset_index()

    flat_train = flat_train.merge(tmp, on = ['month_cnt','bulk_spalen_id'], how = 'left').fillna(0)

    del tmp
    gc.collect()

    flat_train['diff_pricem2_median'] = flat_train['bulk_spalen_pricem2_median']-flat_train['pricem2']
    flat_train['diff_price_median'] = flat_train['bulk_spalen_price_median']-flat_train['price']

    flat_train['diff_pricem2_min'] = flat_train['pricem2'] - flat_train['bulk_spalen_pricem2_min']
    flat_train['diff_price_min'] = flat_train['price'] - flat_train['bulk_spalen_price_min']

    flat_train['diff_pricem2_max'] = flat_train['bulk_spalen_pricem2_max']-flat_train['pricem2']
    flat_train['diff_price_max'] = flat_train['bulk_spalen_price_max']-flat_train['price']


    tmp = flat_train[(flat_train['stat_new']>0) & (flat_train['stat_new']<5)] \
                    .groupby(['month_cnt','bulk_spalen_id']) \
                    .agg({'square':('min','max','std','count','median','mean')})

    tmp = my_drop_levels(tmp, sep = '_', brief = 'bulk_spalen_').reset_index()

    flat_train = flat_train.merge(tmp, on = ['month_cnt','bulk_spalen_id'], how = 'left').fillna(0)

    del tmp
    gc.collect()

    flat_train['diff_square_median'] = flat_train['bulk_spalen_square_median']-flat_train['square']
    flat_train['diff_square_mean'] = flat_train['bulk_spalen_square_mean']-flat_train['square']
    flat_train['diff_square_min'] = flat_train['bulk_spalen_square_min']-flat_train['square']
    flat_train['diff_square_max'] = flat_train['bulk_spalen_square_max']-flat_train['square']



    tmp = flat_train.groupby(['month_cnt','bulk_spalen_id', 'stat_new']) \
                    .size() \
                    .reset_index(name = 'dolya_stat_new')

    tmp1 = flat_train.groupby(['month_cnt','bulk_spalen_id']) \
                    .size() \
                    .reset_index(name = 'cnt_stat_new') 


    tmp1['unique_id'] = tmp1['month_cnt'].astype('str')+'_'+tmp1['bulk_spalen_id'].astype('str')

    tmp = tmp.merge(tmp1) 
    tmp['dolya_stat_new']=tmp['dolya_stat_new']/tmp['cnt_stat_new']


    tmp2 = tmp[['unique_id','stat_new','dolya_stat_new']].pivot(index = 'unique_id', columns='stat_new') \
                                                         .fillna(0) 

    tmp2 = my_drop_levels(tmp2, sep = '_', brief = 'bulk_spalen_').reset_index()
    tmp2 = tmp2.merge(tmp1) 
    #tmp = my_drop_levels(tmp, sep = '_', brief = 'bulk_spalen_').reset_index()

    flat_train = flat_train.merge(tmp2.drop('unique_id', axis = 1), on = ['month_cnt','bulk_spalen_id'], how = 'left').fillna(0) 

    del tmp, tmp1, tmp2
    gc.collect()

    flat_train['month'] = pd.to_datetime(flat_train.date1, format = '%Y-%m-%d').dt.month
    
    flat_train = mem_economy(flat_train)
    gc.collect()
    
    
    tmp = flat_train.groupby(['month_cnt','bulk_spalen_id', 'stat_new']) \
                    .agg({'square':'sum'}) \
                    .reset_index() \
                    .rename(columns = {'square':'dolya_sqr_stat_new'})

    tmp1 = flat_train.groupby(['month_cnt','bulk_spalen_id']) \
                    .agg({'square':'sum'}) \
                    .reset_index() \
                    .rename(columns = {'square':'sqr_stat_new'})


    tmp1['unique_id'] = tmp1['month_cnt'].astype('str')+'_'+tmp1['bulk_spalen_id'].astype('str')

    tmp = tmp.merge(tmp1) 
    tmp['dolya_sqr_stat_new']=tmp['dolya_sqr_stat_new']/tmp['sqr_stat_new']


    tmp2 = tmp[['unique_id','stat_new','dolya_sqr_stat_new']].pivot(index = 'unique_id', columns='stat_new') \
                                                         .fillna(0) 

    tmp2 = my_drop_levels(tmp2, sep = '_', brief = 'bulk_spalen_').reset_index()
    tmp2 = tmp2.merge(tmp1) 
    #tmp = my_drop_levels(tmp, sep = '_', brief = 'bulk_spalen_').reset_index()

    flat_train = flat_train.merge(tmp2.drop('unique_id', axis = 1), on = ['month_cnt','bulk_spalen_id'], how = 'left').fillna(0) 

    del tmp, tmp1, tmp2
    gc.collect()

    return flat_train

def my_simple_cv(model, dataset, study_columns, random_state=442, importance_flag = False):
    
    train_agg = dataset[dataset.is_train==1].copy().reset_index(drop = True)
    test_agg = dataset[dataset.is_train==0].copy().reset_index(drop = True)
    
    ind = 0
    _mse = np.array([],dtype = 'float')
    #заполним нулями предикт теста
    y_test_pred = np.zeros(test_agg.shape[0],dtype = 'float')
    
    
    #основная кросс-валидация
    for train_index, valid_index in KFold(n_splits=5, random_state=random_state, shuffle = True).split(train_agg):   

        tmp_train  = train_agg.loc[train_index,:]
        tmp_valid  = train_agg.loc[valid_index,:]
        tmp_test   = test_agg.copy()

        #учиться будем только на study_columns не на всех переменных     
        X_train = tmp_train.loc[:,study_columns]
        X_valid = tmp_valid.loc[:,study_columns]
        X_test  = tmp_test.loc[:,study_columns]

        y_train = tmp_train['value']
        y_valid = tmp_valid['value']
        y_test = tmp_test['value'] 
        
                
        #обучим модель
        model.fit(X_train,y_train)
 
        y_valid_pred = model.predict(X_valid)
        y_test_pred = y_test_pred+model.predict(X_test)
        
        
        y_valid_pred[y_valid_pred<0] = 0
        
        if ind ==0:
            stacking_df = pd.DataFrame(dict({'bulk_id_int':tmp_valid.bulk_id_int,'predict':y_valid_pred, 'fact':y_valid}))
        else:
            tmp_stacking_df = pd.DataFrame(dict({'bulk_id_int':tmp_valid.bulk_id_int,'predict':y_valid_pred, 'fact':y_valid}))
            stacking_df = stacking_df.append(tmp_stacking_df).sort_values('bulk_id_int')
        
        
        _mse = np.append(_mse,mean_squared_error(y_valid,y_valid_pred))

        ind = ind + 1

        #break
    
    
    importance = pd.DataFrame(dict({'feature':'none', 'delta_mse':0}), index = ['none'])
    
    mse_now = mean_squared_error(y_valid,y_valid_pred)
    NUMBER_SHUFFLE = 5
    if importance_flag:
        for feature in study_columns:

            tmp_mse = 0
            for i in range(NUMBER_SHUFFLE):
                _X_valid = X_valid.copy()
                a = np.asarray(X_valid[feature].copy())
                np.random.shuffle(a)
                _X_valid[feature] = a
                y_valid_pred = model.predict(_X_valid)
                tmp_mse = tmp_mse+mean_squared_error(y_valid, y_valid_pred)/NUMBER_SHUFFLE
            tmp_importance = pd.DataFrame(dict({'feature':feature, 'delta_mse':(tmp_mse-mse_now)}), index = [feature])    
            importance = importance.append(tmp_importance) 
    
    
    #усредняем по фолдам предсказание теста
    y_test_pred = y_test_pred/ind
    
    y_test_pred[y_test_pred<0] = 0
    
    submission = pd.DataFrame(dict({'id':test_agg.id,'value':y_test_pred, 'bulk_spalen_id':test_agg.bulk_spalen_id}))
    
    
    
    return submission, _mse, stacking_df, importance

def my_submit(model, 
                        dataset, 
                        right_dataset, 
                        right_date, 
                        cv_dates,
                        last_date,
                        n_month,
                        study_columns, 
                        value_column, 
                        group_columns, 
                        random_state=442, 
                        importance_flag = False):
    
    #весь обучающий датасет
    train_agg = dataset.copy().reset_index(drop = True)
    
    ind = 0
    _mse = np.array([],dtype = 'float')
    _grp_mse = np.array([],dtype = 'float')
    gc.collect()
    print('==========================')
    
    #основная кросс-валидация
    d = cv_dates[len(cv_dates)-1]
     
    #Расчитаем для submit-а

    #обучающая сдвигается на 1 месяц вперед
    dt = fixed_dates[d+1]
    if d+1-n_month<0:
        dt_start = fixed_dates[0]
    else:
        dt_start = fixed_dates[d+1-n_month]
            
    print('study dataset fot test: date = ',dt,' dt_start = ',dt_start)
        
    tmp_train  = train_agg.loc[(train_agg.date1<dt) & (train_agg.date1>=dt_start),:] 
    #а тестовая - на последнюю известную дату
    tmp_test  = train_agg.loc[train_agg.date1==last_date,:]   
    tmp_right = right_dataset.loc[right_dataset.date1==right_date,:].copy()
        
    #учиться будем только на study_columns не на всех переменных     
    X_train = tmp_train.loc[:,study_columns]
    X_test  = tmp_test.loc[:,study_columns]
    
    print('Максимальная дата обучающей ',tmp_train.date1.max())
    print('Миниимальная дата тестовой ',tmp_test.date1.min())   
    
    y_train = tmp_train[value_column]
        
    del tmp_train
    gc.collect()
        
    #обучим модель
    model.fit(X_train,y_train)
        
    y_test_pred = model.predict(X_test)
    y_test_pred[y_test_pred<0] = 0
        
    R_test = X_test.copy()
    R_test['predict'] = y_test_pred 
             
    R_test = R_test.groupby(group_columns) \
                             .agg({'predict':'sum'}) \
                             .reset_index()
    tmp_right = tmp_right.merge(R_test, on = group_columns, how = 'left')
    submission = tmp_right[['id','bulk_spalen_id','predict']].rename(columns = {'predict':'value'}).fillna(0)
    

    return submission#, _mse, _grp_mse#, importance, model, full_df_for_calc_cv


def my_cv(model, 
                        dataset, 
                        right_dataset, 
                        right_date, 
                        cv_dates,
                        last_date,
                        n_month,
                        study_columns, 
                        value_column, 
                        group_columns, 
                        random_state=442, 
                        importance_flag = False):
    
    #весь обучающий датасет
    train_agg = dataset.copy().reset_index(drop = True)
    
    ind = 0
    _mse = np.array([],dtype = 'float')
    _grp_mse = np.array([],dtype = 'float')
    gc.collect()
    print('==========================')
    
    #основная кросс-валидация
    for d in cv_dates:
        #получаем даты
        dt = fixed_dates[d]
        if d-n_month<0:
            dt_start = fixed_dates[0]
        else:
            dt_start = fixed_dates[d-n_month]
            
        print('ind = ',ind, ' date = ',dt,' dt_start = ',dt_start)
        
        tmp_train  = train_agg.loc[(train_agg.date1<dt) & (train_agg.date1>=dt_start),:]   
        tmp_valid  = train_agg.loc[train_agg.date1==dt,:]
        tmp_right = right_dataset.loc[right_dataset.date1==dt,:].copy()
        
        #учиться будем только на study_columns не на всех переменных     
        X_train = tmp_train.loc[:,study_columns]
        X_valid = tmp_valid.loc[:,study_columns]
        
        y_train = tmp_train[value_column]
        y_valid = tmp_valid[value_column]
        
        del tmp_train#, tmp_valid
        gc.collect()
        
        #обучим модель
        model.fit(X_train,y_train)
        
        y_valid_pred = model.predict(X_valid)
        y_valid_pred[y_valid_pred<0] = 0
        
        R_valid = tmp_valid[['bulk_spalen_id','id_flatwork_int']].copy() #X_valid.copy()
        R_valid['predict'] = y_valid_pred
        
        print(f'X_valid.shape = {X_valid.shape:}')
        _mse = np.append(_mse,mean_squared_error(y_valid,y_valid_pred))

            
        R_valid['value_flat'] = y_valid

        R_valid = R_valid.groupby(group_columns) \
                             .agg({'predict':'sum','value_flat':'sum'}) \
                             .reset_index()
        tmp_right = tmp_right.merge(R_valid, on = group_columns, how = 'left').fillna(0)
            
        if 1==0:
            if ind == 0:
                full_df_for_calc_cv = tmp_right[['value','predict']].copy()
                full_df_for_calc_cv['ind'] = ind
            else:
                tmp_df_for_calc_cv = tmp_right[['value','predict']].copy()
                tmp_df_for_calc_cv['ind'] = ind
                full_df_for_calc_cv = full_df_for_calc_cv.append(tmp_df_for_calc_cv)

            _grp_mse = np.append(_grp_mse,mean_squared_error(tmp_right['value'],tmp_right['predict']))

        ind = ind + 1

        #break
        
        
    #посчитаем важность
    importance = pd.DataFrame(dict({'feature':'none', 'delta_mse':0}), index = ['none'])
    
    mse_now = mean_squared_error(y_valid,y_valid_pred)
    NUMBER_SHUFFLE = 5
    if importance_flag:
        for feature in study_columns:

            tmp_mse = 0
            for i in range(NUMBER_SHUFFLE):
                _X_valid = X_valid.copy()
                a = np.asarray(X_valid[feature].copy())
                np.random.shuffle(a)
                _X_valid[feature] = a
                y_valid_pred = model.predict(_X_valid)
                tmp_mse = tmp_mse+mean_squared_error(y_valid, y_valid_pred)/NUMBER_SHUFFLE
            tmp_importance = pd.DataFrame(dict({'feature':feature, 'delta_mse':(tmp_mse-mse_now)}), index = [feature])    
            importance = importance.append(tmp_importance)     
    
    #Расчитаем для submit-а
    
    #обучающая сдвигается на 1 месяц вперед
    dt = fixed_dates[d+1]
    if d+1-n_month<0:
        dt_start = fixed_dates[0]
    else:
        dt_start = fixed_dates[d+1-n_month]
            
    print('study dataset fot test: date = ',dt,' dt_start = ',dt_start)
        
    tmp_train  = train_agg.loc[(train_agg.date1<dt) & (train_agg.date1>=dt_start),:] 
    #а тестовая - на последнюю известную дату
    tmp_test  = train_agg.loc[train_agg.date1==last_date,:]   
    tmp_right = right_dataset.loc[right_dataset.date1==right_date,:].copy()
        
    #учиться будем только на study_columns не на всех переменных     
    X_train = tmp_train.loc[:,study_columns]
    X_test  = tmp_test.loc[:,study_columns]
        
    y_train = tmp_train[value_column]
        
    del tmp_train
    gc.collect()
        
    #обучим модель
    model.fit(X_train,y_train)
        
    y_test_pred = model.predict(X_test)
    y_test_pred[y_test_pred<0] = 0
        
    R_test = tmp_test[['bulk_spalen_id','id_flatwork_int']].copy() #X_test.copy()
    R_test['predict'] = y_test_pred 
             
    R_test = R_test.groupby(group_columns) \
                             .agg({'predict':'sum'}) \
                             .reset_index()
    tmp_right = tmp_right.merge(R_test, on = group_columns, how = 'left')
    submission = tmp_right[['id','predict']].rename(columns = {'predict':'value'}).fillna(0)
    

    return submission, _mse, _grp_mse, importance, model#, full_df_for_calc_cv


##############################

full = prepare_full()
flat, dict_bulk_spalen, dict_flat = prepare_flat()

full = mem_economy(full)
gc.collect()
flat = mem_economy(flat)
gc.collect()

last_date = full[full.is_train==0].date1.dt.date.astype('str').min()

#Добавим данные о максимальной площади, доступной для продажи
max_sale_square = flat[flat['sale'].astype('str')>last_date].groupby('bulk_spalen_id') \
                                                             .square.sum() \
                                                             .reset_index(name = 'max_square')

full = full.merge(max_sale_square, on = 'bulk_spalen_id', how = 'left').fillna(0)



status, dict_stat = prepare_status()
price             = prepare_price()

#status = mem_economy(status)
#price  = mem_economy(price)

gc.collect()

tmp_calendar = full.loc[full['is_train']==0,('date1','date2')].sort_values('date1').drop_duplicates()
test_days_period =  np.array(((tmp_calendar.date2 - tmp_calendar.date1)/np.timedelta64(1,'D')+1).astype('int32')) 

print(f'test_days_period = {test_days_period:}')
flat_train = prepare_flat_train(test_days_period)

flat_train = mem_economy(flat_train)
gc.collect()

full['calc_last_value'] = full['calc_last_value'].fillna(0)
full = full.reset_index(drop  = True)

column_study = np.array(['До метро пешком(км)', 'price', 'mean_sq', 'price_by_square',
       'mean_fl', 'Cтавка по ипотеке', 'Станций метро от кольца',
       'Площадь двора', 'Date_int', 'До промки(км)', 'month',
       'До большой дороги на машине(км)', 'spalen',
       'Площадь зеленой зоны в радиусе 500 м', 'bulk_id_int',
       'До удобной авторазвязки на машине(км)', 'До парка пешком(км)',
       'Курс', 'До Кремля', 'Вклады свыше 3 лет','calc_last_value'])

lgb_model = lgb.LGBMRegressor(n_estimators = 150, random_state = 42)

submission_lgb, mse, stacking_df, imp_df = my_simple_cv(lgb_model, 
                                                        full, 
                                                        column_study, 
                                                        random_state=442, 
                                                        importance_flag = True)

#rmse на локальной валидации этой модели
rmse = np.sqrt(mse)

submission_lgb = submission_lgb.sort_values('id')
filename = f'c1imb3r_lgb_rmse_{(np.mean(rmse)):.4f} +- {(np.std(rmse)):.4f}.csv'
submission_lgb.to_csv(filename, index = False)



fixed_dates = np.sort(full.date1.dt.strftime('%Y-%m-%d').unique())


gc.collect()
#определим цену продажи
column_filter = ['id_sec','id_gk','id_flatwork','date_settle', 
                 'date_salestart','sale','bulk_id',
                 'date1','realized_1', 'realized_2', 'realized_3',
                 'value_1','value_2', 'value_3','dt_to_sale','flat_startsale','date_flat_startsale']

                
column_study = np.setdiff1d(np.asarray(flat_train.columns), column_filter)


last_date = full[full.is_train==0].date1.dt.date.astype('str').min()

for i in range(1):
    n_month = 15
    lgb_model = lgb.LGBMRegressor(n_estimators = 200, random_state = 42+i, predict_leaf_index = True)
    
    submission_1 = my_submit(
                            model = lgb_model, 
                            dataset = flat_train,
                            right_dataset = full[['id','is_train','bulk_spalen_id','value','date1']],
                            right_date = fixed_dates[-3], 
                            cv_dates = [len(fixed_dates)-4], 
                            last_date = last_date,
                            n_month = n_month,
                            study_columns = column_study, 
                            value_column = 'value_1', 
                            group_columns = 'bulk_spalen_id',
                            random_state=442, 
                            importance_flag = True)

    submission_2 = my_submit(
                            model = lgb_model, 
                            dataset = flat_train,
                            right_dataset = full[['id','is_train','bulk_spalen_id','value','date1']],
                            right_date = fixed_dates[-2], 
                            cv_dates = [len(fixed_dates)-5],
                            last_date = last_date,
                            n_month = n_month,
                            study_columns = column_study, 
                            value_column = 'value_2', 
                            group_columns = 'bulk_spalen_id',
                            random_state=442, 
                            importance_flag = True)

    submission_3 = my_submit(
                            model = lgb_model, 
                            dataset = flat_train,
                            right_dataset = full[['id','is_train','bulk_spalen_id','value','date1']],
                            right_date = fixed_dates[-1],
                            cv_dates = [len(fixed_dates)-6],
                            last_date = last_date,
                            n_month = n_month,
                            study_columns = column_study, 
                            value_column = 'value_3', 
                            group_columns = 'bulk_spalen_id',
                            random_state=442, 
                            importance_flag = True)


    submission_flat = pd.concat([submission_1,submission_2,submission_3]).fillna(0).sort_values('id')
    
    if i==0:
        v = submission_flat['value']
    else: 
        v = v + submission_flat['value']
        
submission_flat['value'] = v/(i+1)  
submission_flat = submission_flat.sort_values('id').reset_index(drop = True)

filename = f'c1imb3r_x_{(i+1):}_nmonth_15.csv'
submission_flat.to_csv(filename, index = False)


best_lgb = submission_lgb.rename(columns = {'value':'predict'})
best_flat = submission_flat.rename(columns = {'value':'predict'})

print(best_lgb.shape)
print(best_flat.shape)

best = best_flat.merge(best_lgb, on = ['id','bulk_spalen_id'], how = 'left')
mic_c = 0.6

best['predict'] = best['predict_x']
best.loc[best['predict_x']==0, 'predict'] = best.loc[best['predict_x']==0, 'predict_y']
best['predict'] = mic_c*best['predict']+(1-mic_c)*best['predict_y']
     
best['value'] = best['predict'] 

filename = f'c1imb3r_submit.csv'
best[['id','value']].to_csv(filename, index = False)


#Добавим знания о квартирах
max_sale_square = flat[(flat['sale'].astype('str')=='2020-01-02') & 
                      (flat['flat_startsale'] < '2018-06-02')].groupby('bulk_spalen_id') \
                                   .agg({'square':'sum'}) \
                                   .reset_index()
res_sale_square = best.groupby('bulk_spalen_id').agg({'predict':'sum'}).reset_index()

res_sale_square = res_sale_square.merge(max_sale_square, on = 'bulk_spalen_id', how = 'left').fillna(0)
res_sale_square['coeff'] = res_sale_square['square']/res_sale_square['predict']
res_sale_square.loc[res_sale_square['coeff']>1, 'coeff'] = 1


best = best.merge(res_sale_square[['bulk_spalen_id','coeff']], how = 'left', on = 'bulk_spalen_id')
best['predict'] = best['predict']*best['coeff']

best['value'] = best['predict'] 

filename = f'c1imb3r_submit_max.csv'
best[['id','value']].to_csv(filename, index = False)


