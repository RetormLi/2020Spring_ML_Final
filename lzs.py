import pandas as pd
import datetime
id_road_2_index = {276183:0, 276184:1, 275911:3, 275912:4, 276240:5, 276241:6, 
    276264:7, 276265:8, 276268:9, 276269:10, 276737:11, 276738:12}
time_2_index = {}
def preprocess_tti(tti_data):
    del tti_data['speed']
    tti_data['id_road'] = tti_data['id_road'].astype(int)
    tti_data['TTI'] = tti_data['TTI'].astype(float)
    tti_data['time'] = pd.to_datetime(
        tti_data['time'], infer_datetime_format=True)
    tti_data['weekday'] = tti_data['time'].dt.dayofweek
    tti_data['hour'] = tti_data['time'].dt.hour
    tti_data['minute'] = tti_data['time'].dt.minute
    tti_data['date'] = tti_data['time'].dt.date.astype(str)
    del tti_data['time']
    tti_data.query('hour > 7 and hour < 21 or hour == 7 and minute > 20 or hour == 21 and minute < 30', inplace=True)
    
    # group = tti_data.groupby('date')
    # print(group.size())
    return tti_data
def preprocess_tti_no_label(tti_data):
    tti_data['id_sample'] = tti_data['id_sample'].astype(int)
    tti_data['id_road'] = tti_data['id_road'].astype(int)
    tti_data['time'] = pd.to_datetime(
        tti_data['time'], infer_datetime_format=True)
    return tti_data
train_TTI = preprocess_tti(pd.read_csv('./data/train_TTI.csv'))
# toPredict_noLabel = preprocess_tti_no_label(pd.read_csv('./data/toPredict_noLabel.csv'))
# toPredict_train_TTI = preprocess_tti(pd.read_csv('./data/toPredict_train_TTI.csv'))
# print(train_TTI)

