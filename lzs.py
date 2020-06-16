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

    # 共12个路段，每个时间段共6个时间点，12*6=72维数据，加上星期几标志weekday，
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

def preprocess_tti_no_label(tti_data):
    tti_data['id_sample'] = tti_data['id_sample'].astype(int)
    tti_data['id_road'] = tti_data['id_road'].astype(int)
    tti_data['time'] = pd.to_datetime(
        tti_data['time'], infer_datetime_format=True)
    tti_data['weekday'] = tti_data['time'].dt.dayofweek
    tti_data['date'] = tti_data['time'].dt.date
    tti_data['time'] = tti_data['time'].dt.time
    return tti_data

generate_time_transfer()

if not os.path.isfile('test_X.csv'):
    print('Generate test data set X')
    toPredict_train_TTI = preprocess_tti(pd.read_csv('./data/toPredict_train_TTI.csv'), filter=False)
    test_X, _ = generate_Train(toPredict_train_TTI, include_date=True)
    test_X = test_X.dropna()
    test_X.to_csv('test_X.csv', na_rep='NaN', index=False)
else:
    test_X = pd.read_csv('test_X.csv')
test_X, date_info = test_X.iloc[:, :-1], test_X['date']

if not os.path.isfile('train_X.csv') or not os.path.isfile('train_y.csv'):
    print('Generate train data set, might be slow.')
    train_TTI = preprocess_tti(pd.read_csv('./data/train_TTI.csv'))
    train_X, train_y = generate_Train(train_TTI)
    train_X.to_csv('train_X.csv', na_rep='NaN', index=False)
    train_y.to_csv('train_y.csv', na_rep='NaN', index=False)
else:
    train_X, train_y = pd.read_csv('train_X.csv'), pd.read_csv('train_y.csv')
# print(train_X)
# print(train_y)
# print(test_X)
# print(date_info)
# import lightgbm as lgb
# from sklearn.metrics import mean_squared_error
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split

# # 转换为Dataset数据格式
# lgb_train = lgb.Dataset(X_train, y_train)
# lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# # 参数
# params = {
#     'task': 'train',
#     'boosting_type': 'gbdt',  # 设置提升类型
#     'objective': 'regression',  # 目标函数
#     'metric': {'l2', 'auc'},  # 评估函数
#     'num_leaves': 31,  # 叶子节点数
#     'learning_rate': 0.05,  # 学习速率
#     'feature_fraction': 0.9,  # 建树的特征选择比例
#     'bagging_fraction': 0.8,  # 建树的样本采样比例
#     'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
#     'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
# }

# # 模型训练
# gbm = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=lgb_eval, early_stopping_rounds=5)

# # 模型保存
# gbm.save_model('model.txt')

# # 模型加载
# gbm = lgb.Booster(model_file='model.txt')

# # 模型预测
# y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

# # 模型评估
# print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

# print(train_X)
# print(train_y)

# print(train_TTI)

