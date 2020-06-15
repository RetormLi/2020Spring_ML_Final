import pandas as pd
import datetime


def time_day(dateTime):
    daylist = dateTime.split()
    weekday = (int(datetime.datetime.strptime(daylist[0],'%Y/%m/%d').strftime("%w"))+1)/7
    time = (int(daylist[1].split(":")[0])*3600 + int(daylist[1].split(":")[1])*60 + int(daylist[1].split(":")[2]))/86400
    return daylist[0], weekday, time  


def gps_str_to_list(gps_data):
    gps_list = gps_data[1:-1].split(',')
    gps_list = [i.split() for i in gps_list]
    for i in range(len(gps_list)):
        # 去除方向数据, 切分时间数据
        day, weekday, time = time_day(pd.to_datetime(gps_list[i][4], unit='s', utc=True).tz_convert(
            "Asia/Shanghai").to_period("S").strftime('%Y/%m/%d %H:%M:%S'))
        gps_list[i] = [float(gps_list[i][0]),
                       float(gps_list[i][1]),
                       round(float(gps_list[i][2])*3.6, 1),
                       day,  # 日期
                       weekday,  # 周日-周六：1/7-7/7
                       time] # 归一化的当日时间
    return gps_list


def tti_time_transfer(dateTime):
    daylist = dateTime.split()
    weekday = (int(datetime.datetime.strptime(daylist[0],'%Y-%m-%d').strftime("%w"))+1)/7
    time = (int(daylist[1].split(":")[0])*3600 + int(daylist[1].split(":")[1])*60)/86400
    return daylist[0], weekday, time  


def preprocess_tti(tti_data):
    tti_data['id_road'] = tti_data['id_road'].astype(str)
    tti_data['TTI'] = tti_data['TTI'].astype(float)
    tti_data['speed'] = tti_data['speed'].astype(float)
    tti_data['time'] = pd.to_datetime(
        tti_data['time'], infer_datetime_format=True)
    tti_data['day'] = ''  # 日期
    tti_data['weekday'] = ''  # 周日-周六：1/7-7/7
    tti_data['daytime'] = ''  # 归一化的当日时间
    for i in range(len(tti_data['time'])):
        tti_data['day'][i], tti_data['weekday'][i], tti_data['daytime'][i] = tti_time_transfer(str(tti_data['time'][i]))
    tti_data = tti_data.drop(columns = 'time')
    return tti_data

