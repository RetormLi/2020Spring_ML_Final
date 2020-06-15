import pandas as pd
import datetime


def time_day(dateTime):
    daylist = dateTime.split()
    weekday = int(datetime.datetime.strptime(daylist[0],'%Y/%m/%d').strftime("%w"))
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
                       weekday,  # 周日-周六：0-6
                       time] # 归一化的当日时间
    return gps_list

