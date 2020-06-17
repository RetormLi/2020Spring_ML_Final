import pandas as pd
import numpy as np
from datetime import time, timedelta, date, datetime
from collections import defaultdict
import os.path

id_roads = [276183, 276184, 275911, 275912, 276240, 276241, 
276264, 276265, 276268, 276269, 276737, 276738]

time_labels_begin = [(8, 30), (9, 0), (10, 30), (11, 0), (12, 30), (13, 0), (14, 30), 
(15, 0), (16, 30), (17, 0), (18, 30), (19, 0), (20, 30), (21, 0)]

feature_time = defaultdict(list)
label_time = {}

def add_time(x, delta):
    return (datetime.combine(date.min, x) + delta).time()

# 将时间数据翻译到对应feature或label的时间段和时间点。
def generate_time_transfer():
    for time_slot, (hour, minute) in enumerate(time_labels_begin):
        period = timedelta(minutes=10)
        begin_label = time(hour, minute)
        begin_feature = time(hour - 1, minute)
        for time_point in range(3):
            label_time[begin_label] = (time_slot, time_point)
            begin_label = add_time(begin_label, period)
        for feature_time_point in range(6):
            feature_time[begin_feature].append((time_slot, feature_time_point))
            begin_feature = add_time(begin_feature, period)


def preprocess_tti(tti_data, filter=True):
    del tti_data['speed']
    tti_data['id_road'] = tti_data['id_road'].astype(int)
    tti_data['TTI'] = tti_data['TTI'].astype(float)
    tti_data['time'] = pd.to_datetime(
        tti_data['time'], infer_datetime_format=True)
    tti_data['weekday'] = tti_data['time'].dt.dayofweek
    tti_data['date'] = tti_data['time'].dt.date
    tti_data['time'] = tti_data['time'].dt.time
    if filter:
        begin, end = time(7, 30), time(21, 20)
        tti_data = tti_data[(tti_data['time'] >= begin) & (tti_data['time'] <= end)]
    return tti_data

def generate_Train(tti_data, include_date=False):

    # 共12个路段，每个时间段共6个特征时间点，12*6=72维数据，加上星期几标志weekday，
    # 时间段标志time_slot，预测时间点标志time_point，预测路段标志id_road，处理每日数据时作为Index方便查找
    columns = [str(id_road) + '_' + str(feature_time_point) for id_road in id_roads for feature_time_point in range(6)] + ['weekday']#, 'time_slot', 'time_point', 'id_road']
    
    # 每天共12个路段，每天共14个预测时间段，每个时间段共3个时间点，共12*14*3=504个样本（行）
    index = pd.MultiIndex.from_product([range(14), range(3), id_roads], names=['time_slot', 'time_point', 'id_road'])
    
    train_X = pd.DataFrame(columns=columns)
    train_y = pd.Series(dtype='float64', name='label')

    for (date, weekday), daily_tti_data in tti_data.groupby(['date', 'weekday']):

        del daily_tti_data['date']
        del daily_tti_data['weekday']
        daily_trian_X = pd.DataFrame(index=index, columns=columns)
        daily_trian_X['weekday'] = weekday
        if include_date:
            daily_trian_X['date'] = date

        daily_trian_y = pd.Series(index=index, dtype='float64', name='label')

        for row in daily_tti_data.itertuples():
            time = row.time
            id_road = row.id_road
            TTI = row.TTI
            if time in feature_time:
                for time_slot, feature_time_point in feature_time[time]:
                    daily_trian_X.loc[time_slot, str(id_road) + '_' + str(feature_time_point)] = TTI
            if time in label_time:
                time_slot, time_point = label_time[time]
                daily_trian_y.loc[time_slot, time_point, id_road] = TTI

        train_X = train_X.append(daily_trian_X.reset_index())
        train_y = train_y.append(daily_trian_y.reset_index(drop=True))
    train_X['weekday'] = train_X['weekday'].astype(int)
    train_X['time_slot'] = train_X['time_slot'].astype(int)
    train_X['time_point'] = train_X['time_point'].astype(int)
    train_X['id_road'] = train_X['id_road'].astype(int)
    return train_X, train_y



generate_time_transfer()

if not os.path.isfile('test_X.csv'):
    print('Generate test data set X')
    toPredict_train_TTI = preprocess_tti(pd.read_csv('./data/toPredict_train_TTI.csv'), filter=False)
    # 保留日期数据，为了合并
    test_X, _ = generate_Train(toPredict_train_TTI, include_date=True)
    # 有冗余。去除不需要预测的时间段。
    test_X = test_X.dropna()
    test_X.to_csv('test_X.csv', na_rep='NaN', index=False)
else:
    test_X = pd.read_csv('test_X.csv')

# 保留日期数据，为了合并
test_X, date_info = test_X.iloc[:, :-1], test_X['date']

if not os.path.isfile('train_X.csv') or not os.path.isfile('train_y.csv'):
    print('Generate train data set, might be slow.')
    train_TTI = preprocess_tti(pd.read_csv('./data/train_TTI.csv'))
    train_X, train_y = generate_Train(train_TTI)
    train_X.to_csv('train_X.csv', na_rep='NaN', index=False)
    train_y.to_csv('train_y.csv', na_rep='NaN', index=False)
else:
    train_X, train_y = pd.read_csv('train_X.csv'), pd.read_csv('train_y.csv')

# 去掉NaN样本或标签
train_X['label'] = train_y
train_X.dropna(inplace=True)
train_X, train_y = train_X.iloc[:, :-1], train_X['label']

def categorize(data):
    if 'weekday' in data:
        data['weekday'] = data['weekday'].astype('category')
    data['time_slot'] = data['time_slot'].astype('category')
    data['time_point'] = data['time_point'].astype('category')
    data['id_road'] = data['id_road'].astype('category')
    return data
train_X, test_X = categorize(train_X), categorize(test_X)

import lightgbm as lgbm
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV

params = {
    'objective': 'regression_l1',
    'colsample_bytree': 0.8, 
    'learning_rate': 0.05, 
    'max_depth': 4,
    'n_jobs' : -1,
    'num_leaves' : 15,
    'n_estimators': 6000,
    'colsample_bytree': 0.8,
    'subsample' : 0.8,
    'subsample_freq' : 2
}
# Best score: 0.6483964125096509
# 网格搜索，参数优化
# estimator = LGBMRegressor(**params)
# param_grid = {
#     'n_estimators' : [4000, 5000, 6000]
# }
# search = GridSearchCV(estimator, param_grid, n_jobs=-1)
# print('Find best params, could be really slow.')
# search.fit(train_X, train_y)
# print('Best parameters found by grid search are:', search.best_params_)
# print('Best score:', search.best_score_)
# Best parameters found by grid search are: {'colsample_bytree': 0.8, 'learning_rate': 0.05, 'max_depth': 8, 'n_estimators': 300, 'num_leaves': 63, 'subsample': 0.6}



if not os.path.isfile('model.txt'):
    train_set = lgbm.Dataset(data=train_X, label=train_y)
    model = lgbm.train(params=params, train_set=train_set, categorical_feature=['weekday','time_slot','time_point','id_road'])
    model.save_model('model.txt')
else:
    model = lgbm.Booster(model_file='model.txt')
test_y = model.predict(test_X)
test_X['label'] = test_y
def preprocess_tti_no_label(tti_data):
    # tti_data['id_sample'] = tti_data['id_sample'].astype(int)
    tti_data['id_road'] = tti_data['id_road'].astype(int)
    tti_data['time'] = pd.to_datetime(
        tti_data['time'], infer_datetime_format=True)
    tti_data['date'] = tti_data['time'].dt.date
    tti_data['time'] = tti_data['time'].dt.time
    tti_data[['time_slot', 'time_point']] = tti_data.apply(lambda row: label_time[row['time']], axis=1, result_type="expand")
    del tti_data['time']
    return categorize(tti_data)
test_X['date'] = date_info
test_y = test_X[['date', 'time_slot', 'time_point', 'id_road', 'label']]
noLabel = preprocess_tti_no_label(pd.read_csv('./data/toPredict_noLabel.csv'))
noLabel['date'] = noLabel['date'].astype(str)
test_y['date'] = test_y['date'].astype(str)
result = noLabel.merge(test_y, on=['id_road', 'date', 'time_slot', 'time_point'])
result = result[['id_sample', 'label']]
result.rename(columns={'label' : 'TTI'}, inplace=True)
result.to_csv('result.csv', index=False)