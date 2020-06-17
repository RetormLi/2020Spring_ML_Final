import pandas as pd
import datetime
from ast import literal_eval


def gps_str_to_list(gps_data):
    gps_list = gps_data[1:-1].split(',')
    gps_list = [i.split() for i in gps_list]
    for i in range(len(gps_list)):
        # 去除方向数据
        daylist = pd.to_datetime(gps_list[i][4], unit='s', utc=True).tz_convert(
            "Asia/Shanghai").to_period("S").strftime('%Y/%m/%d %H:%M:%S').split()
        day = int(datetime.datetime.strptime(daylist[0],'%Y/%m/%d').strftime("%w"))
        time = (int(daylist[1].split(":")[0])*3600 + int(daylist[1].split(":")[1])*60 + int(daylist[1].split(":")[2]))/86400
        gps_list[i] = [float(gps_list[i][0]),
                       float(gps_list[i][1]),
                       round(float(gps_list[i][2])*3.6, 1),
                       float(gps_list[i][3]),
                       str(daylist[1].split(":")[0])+":"+str(daylist[1].split(":")[1]),
                       daylist[0],  # 日期
                       day,  # 周日-周六：0-6
                       time] # 归一化的当日时间
    return gps_list


def preprocess_tti(tti_data):
    tti_data['id_road'] = tti_data['id_road'].astype(str)
    tti_data['TTI'] = tti_data['TTI'].astype(float)
    tti_data['speed'] = tti_data['speed'].astype(float)
    tti_data['time'] = pd.to_datetime(
        tti_data['time'])#, infer_datetime_format=True)
    tti_data['day'] = tti_data['time'].apply(lambda x : str(x)[:-3])
    tti_data['weekday'] = (tti_data['time'].apply(lambda x : int(datetime.datetime.strptime(str(x).split()[0],'%Y-%m-%d').strftime("%w")))+1)/7
    tti_data['daytime'] = (tti_data['time'].apply(lambda x : int(str(x).split()[1].split(":")[0]))*3600 + tti_data['time'].apply(lambda x : int(str(x).split()[1].split(":")[1]))*60 + tti_data['time'].apply(lambda x : int(str(x).split()[1].split(":")[2])))/86400
    tti_data = tti_data.drop(columns = 'time')
    return tti_data


roads = {('276183', '276184'): [(114.018188,22.588653),(114.027565,22.591505),(114.027758,22.590376),(114.017544,22.586295)],
        ('275911', '275912'): [(114.01726,22.603209),(114.017045,22.58946),(114.014041,22.589064),(114.015629,22.602099)],
         ('276240', '276241'): [(114.016444,22.606378),(114.025585,22.612479), (114.027516,22.610934),(114.017431,22.604001)],
         ('276264', '276265'): [(114.027677,22.592268), (114.026433,22.60463),(114.027968,22.604453), (114.034105,22.593478)],
         ('276268', '276269'): [(114.035053,22.603313),(114.022179,22.61603),(114.023552,22.61706),(114.036341,22.603828)],
         ('276738', '276737'): [(114.030436,22.602313), (114.025071,22.608316),(114.025994,22.608969),(114.03123,22.602729)]}

# 1：西往东：[(0, 180)]
# 2：东往西：[(180, 360)]
# 3：南往北：[(0, 90), (270, 360)]
# 4：北往南：[(90, 270)]
roads_direction = {'276183': 1, '276184': 2, '275911': 3, '275912': 4, '276240': 1, '276241': 2, '276264': 3, '276265': 4, '276268': 3,'276269': 4,
    '276738': 3, '276737': 4}

def get_road(direction, roads):
    directions = [roads_direction[roads[0]], roads_direction[roads[1]]]
    if directions == [1, 2]:
        if 0 <= direction < 180:
            return roads[0]
        else:
            return roads[1]
    else:
        if 90 <= direction < 270:
            return roads[1]
        else:
            return roads[0]

def get_block_rec(block):
    # 上下左右
    block_rec = [max([i[1] for i in block]), min([i[1] for i in block]),min([i[0] for i in block]),max([i[0] for i in block])]
    return block_rec


def in_block(sample, block):
    block_rec = get_block_rec(block)
    # 初筛，上下左右
    x, y = sample
    if y > block_rec[0] or \
            y < block_rec[1] or \
            x < block_rec[2] or \
            x > block_rec[3]:
        return False
    # 细筛，计算四边形
    A, B, C, D = block
    # 顺时针的格子顶点，若逆时针方向则false
    a = (B[0]-A[0])*(y-A[1])-(B[1]-A[1])*(x-A[0])
    b = (C[0]-B[0])*(y-B[1])-(C[1]-B[1])*(x-B[0])
    c = (D[0]-C[0])*(y-C[1])-(D[1]-C[1])*(x-C[0])
    d = (A[0]-D[0])*(y-D[1])-(A[1]-D[1])*(x-D[0])
    if (a > 0 and b > 0 and c > 0 and d > 0) or \
            (a < 0 and b < 0 and c < 0 and d < 0):
        return True
    return False


def cars(x, y): 
    for i in range(len(x)): # i是每条轨迹
        j = 0
        temp1 = '' # 后续会放入上一个点的路段
        temp2 = '' # 后续会放入上一个点的所属时间（每10分钟为一个时间区间）
        while j < len(x[i]): # j是每条轨迹上的每个点
            gps = (x[i][j][0], x[i][j][1]) # 第j个点的gps
            for block in roads.keys():
                if in_block(gps, roads[block]):
                    road = get_road(x[i][j][3], block) # 第j个点的路段
                    day = x[i][j][5][:4]+"-"+x[i][j][5][5:7]+"-"+x[i][j][5][8:]+" "+x[i][j][4][:-1]+"0" 
                    # 第j个点的所属时间（日期-小时-十分，如2019-12-21 08:30）
                    if (road != temp1 or day != temp2) and temp1 != '' and temp2 != '': 
                    # 若第j个点的路段或时间和前一个点不一样，说明该车换了一条路或者时间过了，那就需要将车在刚才时间走的那条路段的车辆数+1，后面and temp是排除第一个点）
                        idx = [i for i in y[y['id_road'].values == temp1].index if i in y[y['day'].values == day].index] 
                        # 前一个点的路段和时间（即车刚才走的那条路）在tti表中的序号
                        y.loc[idx[0], 'count'] += 1 # 该序号的车辆数+1
                    temp1 = road 
                    temp2 = day
            j += 10 # 每次跳过10个点
    return y


g1 = pd.read_csv('pred_gps_0-10000-1.csv') #g1就是用gps_str_to_list跑出来的前1w条测试集的数据
g1['gps_records'] = g1['gps_records'].apply(literal_eval)

x = g1['gps_records'].tolist()

tti_pred_path = "D:/third/ML/final/traffic1/toPredict_train_TTI.csv"
tti_pred_data = pd.read_csv(tti_pred_path)
tti_pred_data = preprocess_tti(tti_pred_data)
tti_pred_data['count'] = 0


tti_pred_data = cars(x, tti_pred_data)
tti_pred_data.to_csv("tti_pred_0-10000.csv")


