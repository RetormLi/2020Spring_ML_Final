
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


def generate_time_transfer():
    # 将时间数据翻译到对应feature或label的时间段和时间点。
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
        tti_data = tti_data[(tti_data['time'] >= begin)
                            & (tti_data['time'] <= end)]
    return tti_data


def time_to_fraction(datetime):
    second = datetime.hour * 3600 + datetime.minute * 60
    return second / 86400


def generate_time_Train(tti_data):
    road_train_X = []
    road_train_Y = []
    sample = []
    init = False
    for row in tti_data.itertuples():
        time = row.time
        weekday = row.weekday
        TTI = round(row.TTI, 5)
        if len(sample) == 6:
            init = True
        if not init:
            sample.append(TTI)
        else:
            sample.append(TTI)
            road_train_X.append(
                sample[:-1]+[time_to_fraction(time), weekday/7])
            road_train_Y.append(sample[-1])
            sample = sample[1:]
    return np.array(road_train_X), np.array(road_train_Y)


def generate_Train(tti_data):
    road_train_X = []
    road_train_Y = []
    sample = []
    init = False
    for row in tti_data.itertuples():
        time = row.time
        weekday = row.weekday
        TTI = round(row.TTI, 5)
        if len(sample) == 6:
            init = True
        if not init:
            sample.append(TTI)
        else:
            sample.append(TTI)
            road_train_X.append(
                sample[:-1])
            road_train_Y.append(sample[-1])
            sample = sample[1:]
    return np.array(road_train_X), np.array(road_train_Y)


def preprocess_tti_no_label(tti_data):
    # tti_data['id_sample'] = tti_data['id_sample'].astype(int)
    tti_data['id_road'] = tti_data['id_road'].astype(int)
    tti_data['time'] = pd.to_datetime(
        tti_data['time'], infer_datetime_format=True)
    tti_data['date'] = tti_data['time'].dt.date
    tti_data['time'] = tti_data['time'].dt.time
    tti_data[['time_slot', 'time_point']] = tti_data.apply(
        lambda row: label_time[row['time']], axis=1, result_type="expand")
    del tti_data['time']
    return tti_data


generate_time_transfer()

path = 'E:/2020Spring/MachineLearning/2020Spring_ML_Final/'

if not os.path.isfile('new_data/276183_knn_train_X.csv'):

    print('Generate train data set, might be slow.')
    train_TTI = preprocess_tti(pd.read_csv(
        'G:/data/datas/traffic1/train_TTI.csv'))
    train_X, train_y = generate_time_Train(train_TTI)
    road_ttis = dict()
    for road, road_tti in train_TTI.groupby(['id_road']):
        del road_tti['id_road']
        del road_tti['date']
        road_ttis[str(road)] = pd.DataFrame(road_tti)

    train_X = []
    train_Y = []
    for road in id_roads:
        road = str(road)
        road_train_X, road_train_Y = generate_time_Train(road_ttis[road])
        assert (len(road_train_X) == len(road_train_Y))
        assert (len(road_train_X[-1]) == 8)
        train_X.extend(road_train_X)
        train_Y.extend(road_train_Y)
        np.savetxt('new_data/'+road+'_knn_train_X.csv',
                   road_train_X, fmt='%f')
        np.savetxt('new_data/'+road + '_knn_train_Y.csv',
                   road_train_Y, fmt='%f')

# np.savetxt('data/all_knn_train_X.csv',
#            np.array(train_X), fmt='%f')
# np.savetxt('data/all_knn_train_Y.csv',
#            np.array(train_Y), fmt='%f')
