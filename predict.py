#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
import random
import pandas as pd

id_roads = ['276183', '276184', '275911', '275912', '276240', '276241',
            '276264', '276265', '276268', '276269', '276737', '276738']

kNNs = dict()
for road in id_roads:
    train_X = np.loadtxt('data/' + road + '_knn_train_X.csv', dtype='float')
    train_Y = np.loadtxt('data/' + road + '_knn_train_Y.csv', dtype='float')

    neigh = KNeighborsRegressor(n_neighbors=7, weights='distance')
    neigh.fit(train_X, train_Y)
    kNNs[road] = neigh


def begin_time(target_time):
    begin = target_time[:]
    clock = int(begin[11:13])
    clock -= 1
    if clock < 10:
        str_clock = '0'+str(clock)
    else:
        str_clock = str(clock)
    begin = begin[:11]+str_clock+begin[13:]
    return begin


toPredict_train_TTI = pd.read_csv(
    'G:/data/datas/traffic1/toPredict_train_TTI.csv')
toPredict_noLabel = pd.read_csv('G:/data/datas/traffic1/toPredict_noLabel.csv')
target = toPredict_noLabel.values

pred_y = []
for index in range(0, target.shape[0], 3):
    road = target[index][1]
    model = kNNs[str(road)]
    begin = begin_time(target[index][2])
    begin_index = int(toPredict_train_TTI[toPredict_train_TTI['id_road'].isin(
        [road]) & toPredict_train_TTI['time'].isin([begin])].index.values)
    # 7:30 - 8:20
    six_samples = list(
        toPredict_train_TTI.loc[begin_index: begin_index + 5].TTI.values)
    # 8:30
    x_sample1 = model.predict(np.array(six_samples).reshape(1, -1))
    pred_y.append(x_sample1)
    six_samples = six_samples[1:] + [x_sample1]
    # 8:40
    x_sample2 = model.predict(np.array(six_samples).reshape(1, -1))
    pred_y.append(x_sample2)
    six_samples = six_samples[1:] + [x_sample1]
    # 8:50
    x_sample3 = model.predict(np.array(six_samples).reshape(1, -1))
    pred_y.append(x_sample3)

data_dict = {'id_sample': [i for i in range(len(pred_y))], 'TTI': pred_y}
df = pd.DataFrame(data_dict)
df['id_sample'] = df['id_sample'].astype(str)
df['TTI'] = df['TTI'].astype(float)
df.to_csv('prediction.csv', index=False)
