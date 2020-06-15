import pandas as pd
import numpy as np
from datetime import time, timedelta, date, datetime
from collections import defaultdict
def preprocess_tti(tti_data):
    del tti_data['speed']
    tti_data['id_road'] = tti_data['id_road'].astype(str)
    tti_data['TTI'] = tti_data['TTI'].astype(float)
    tti_data['time'] = pd.to_datetime(
        tti_data['time'], infer_datetime_format=True)
    tti_data['weekday'] = tti_data['time'].dt.dayofweek
    tti_data['date'] = tti_data['time'].dt.date
    tti_data['time'] = tti_data['time'].dt.time
    begin, end = time(7, 30), time(21, 20)
    tti_data = tti_data[(tti_data['time'] >= begin) & (tti_data['time'] <= end)]
    return tti_data
# id_roads = ['276183', '276184', '275911', '275912', '276240', '276241', 
# '276264', '276265', '276268', '276269', '276737', '276738']
id_roads = [276183, 276184, 275911, 275912, 276240, 276241, 
276264, 276265, 276268, 276269, 276737, 276738]
time_labels_begin = [(8, 30), (9, 0), (10, 30), (11, 0), (12, 30), (13, 0), (14, 30), 
(15, 0), (16, 30), (17, 0), (18, 30), (19, 0), (20, 30), (21, 0)]
feature_time = defaultdict(list)
label_time = {}
def add_time(x, delta):
    return (datetime.combine(date.min, x) + delta).time()
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
# print(label_time)
# print(feature_time)
# exit(0)
def generate_Train(tti_data):
    # 共12个路段，每个时间段共6个时间点，12*6=72维数据，加上时间段标志time_slot，星期几标志weekday，预测时间点标志time_point，预测路段标志id_road。
    columns = [str(id_road) + '_' + str(feature_time_point) for id_road in id_roads for feature_time_point in range(6)] + ['weekday', 'time_slot', 'time_point', 'id_road']
    # 每天共12个路段，每天共14个预测时间段，每个时间段共3个时间点，共12*14*3=504个样本（行）
    train_X = pd.DataFrame(columns=columns)
    train_y = pd.Series(dtype='float64')
    daily_data = [[np.nan] * 73 + [time_slot, time_point, id_road] for time_slot in range(14) for time_point in range(3) for id_road in id_roads]
    for (_, weekday), daily_tti_data in tti_data.groupby(['date', 'weekday']):
        del daily_tti_data['date']
        del daily_tti_data['weekday']
        daily_trian_X = pd.DataFrame(data=daily_data, columns=columns)
        daily_trian_X['weekday'] = weekday
        daily_trian_y = pd.Series(index=daily_trian_X.index, dtype='float64')
        # TODO: fill in the TTI blanks of daily_trian_X and daily_trian_y using daily_tti_data!
        
        train_X = train_X.append(daily_trian_X)
        train_y = train_y.append(daily_trian_y)
    return train_X, train_y
# def preprocess_tti_no_label(tti_data):
#     tti_data['id_sample'] = tti_data['id_sample'].astype(int)
#     tti_data['id_road'] = tti_data['id_road'].astype(int)
#     tti_data['time'] = pd.to_datetime(
#         tti_data['time'], infer_datetime_format=True)
#     return tti_data
train_TTI = preprocess_tti(pd.read_csv('./data/train_TTI.csv'))
train_X, train_y = generate_Train(train_TTI)
# print(train_X)
# print(train_y)
# toPredict_noLabel = preprocess_tti_no_label(pd.read_csv('./data/toPredict_noLabel.csv'))
# toPredict_train_TTI = preprocess_tti(pd.read_csv('./data/toPredict_train_TTI.csv'))
# print(train_TTI)

