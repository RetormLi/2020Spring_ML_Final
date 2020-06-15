#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import pandas as pd
import datetime


def gps_time_transfer(dateTime):
    daylist = dateTime.split()
    weekday = (int(datetime.datetime.strptime(
        daylist[0], '%Y/%m/%d').strftime("%w"))+1)/7
    time = (int(daylist[1].split(":")[
            0])*3600 + int(daylist[1].split(":")[1])*60 + int(daylist[1].split(":")[2]))/86400
    return daylist[0], weekday, time


def gps_str_to_list(gps_data):
    gps_list = gps_data[1:-1].split(',')
    gps_list = [i.split() for i in gps_list]
    for i in range(len(gps_list)):
        # 切分时间数据
        day, weekday, time = gps_time_transfer(pd.to_datetime(gps_list[i][4], unit='s', utc=True).tz_convert(
            "Asia/Shanghai").to_period("S").strftime('%Y/%m/%d %H:%M:%S'))
        gps_list[i] = [float(gps_list[i][0]),
                       float(gps_list[i][1]),
                       round(float(gps_list[i][2]) * 3.6, 1),
                       float(gps_list[i][3]),
                       day,  # 日期
                       weekday,  # 周日-周六：1/7-7/7
                       time]  # 归一化的当日时间
    return gps_list


def preprocess_gps(gps_data):
    gps_data['id_order'] = gps_data['id_order'].astype(str)
    gps_data['id_user'] = gps_data['id_user'].astype(str)
    gps_data['gps_records'] = gps_data['gps_records'].map(
        gps_str_to_list)
    return gps_data


def preprocess_tti(tti_data):
    tti_data['id_road'] = tti_data['id_road'].astype(str)
    tti_data['TTI'] = tti_data['TTI'].astype(float)
    tti_data['speed'] = tti_data['speed'].astype(float)
    tti_data['time'] = pd.to_datetime(
        tti_data['time'], infer_datetime_format=True)
    return tti_data


gps_train_path = "G:/data/datas/20191201_20191220.csv"
tti_train_path = "G:/data/datas/traffic1/train_TTI.csv"

gps_NDATA = 2
tti_NDATA = 2

gps_train_data = pd.read_csv(gps_train_path, nrows=gps_NDATA, header=None)
gps_train_data.columns = ['id_order', 'id_user', 'gps_records']
gps_train_data = preprocess_gps(gps_train_data)
tti_train_data = pd.read_csv(tti_train_path, nrows=tti_NDATA)
tti_train_data = preprocess_tti(tti_train_data)
