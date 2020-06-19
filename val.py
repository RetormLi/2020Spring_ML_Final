#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
import random

id_roads = ['276183', '276184', '275911', '275912', '276240', '276241',
            '276264', '276265', '276268', '276269', '276737', '276738']


def val_with_road():
    count = 0
    sum_mae = 0
    for road in id_roads:
        print(road)
        X = np.loadtxt('new_data/'+road+'_knn_train_X.csv', dtype='float')
        Y = np.loadtxt('new_data/'+road+'_knn_train_Y.csv', dtype='float')

        RATIO = 0.2
        m = X.shape[0]
        for i in range(10):
            count += 1
            index_all = np.array(range(m))
            index_valid = random.sample(range(m), int(RATIO * m))
            index_train = np.delete(index_all, index_valid, axis=0)
            train_X = X[index_train]
            train_y = Y[index_train]
            val_X = X[index_valid]
            val_y = Y[index_valid]

            # neigh = KNeighborsRegressor(n_neighbors=3)
            neigh = KNeighborsRegressor(n_neighbors=7, weights='distance')
            neigh.fit(train_X, train_y)
            pred_y = neigh.predict(val_X)
            mae = mean_absolute_error(pred_y, val_y)
            # print(mae)
            sum_mae += mae
    print(sum_mae/count)


def val_all():
    count = 0
    sum_mae = 0
    X = np.loadtxt('olddata/all_knn_train_X.csv', dtype='float')
    Y = np.loadtxt('olddata/all_knn_train_Y.csv', dtype='float')

    RATIO = 0.2
    m = X.shape[0]
    for i in range(10):
        count += 1
        index_all = np.array(range(m))
        index_valid = random.sample(range(m), int(RATIO * m))
        index_train = np.delete(index_all, index_valid, axis=0)
        train_X = X[index_train]
        train_y = Y[index_train]
        val_X = X[index_valid]
        val_y = Y[index_valid]

        # neigh = KNeighborsRegressor(n_neighbors=3)
        neigh = KNeighborsRegressor(n_neighbors=5, weights='distance')
        neigh.fit(train_X, train_y)
        pred_y = neigh.predict(val_X)
        mae = mean_absolute_error(pred_y, val_y)
        # print(mae)
        sum_mae += mae
    print(sum_mae/count)


# val_all()
val_with_road()
