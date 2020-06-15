# import lightgbm as lgb
# from sklearn.metrics import mean_squared_error
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
 
# iris = load_iris()
# data = iris['data']
# target = iris['target']
# X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
# print(X_train, y_train)
# print(X_test, y_test)
# # 创建成lgb特征的数据集格式
# lgb_train = lgb.Dataset(X_train, y_train)
# lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
 
# # 将参数写成字典下形式
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
 
# # 训练 cv and train
# gbm = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=lgb_eval, early_stopping_rounds=5)
 
# # 保存模型到文件
# gbm.save_model('model.txt')
 
# # 预测数据集
# y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
 
# # 评估模型
# print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

import pandas as pd
 
train_sales = pd.read_csv('./Train/train_sales_data.csv',header=0)
train_search = pd.read_csv('./Train/train_search_data.csv',header=0)
data=train_sales.merge(train_search,on=("adcode","model","regYear","regMonth"),how='inner')
data=data.drop(['province_x','province_y'], axis=1)
print(data)
 
 
import copy
categoricals = ['model', 'adcode','bodyType']
for feature in categoricals:
    df = copy.copy(pd.get_dummies(data[feature], drop_first=True))
    data= pd.concat([data, df], axis=1)
    data.drop(columns=feature, inplace=True)
print(data.head())  
 
 
def to_supervised(data):
    x = data.iloc[0:1320*20,:].values
    y = data.iloc[1320*4:1320*24,2].values
    return x, y
 
data_x,data_y=to_supervised(data)
print(data_x.shape)
print(data_y.shape)
train_x,test_x=data_x[0:1320*16],data_x[1320*16:26399+1]
train_y,test_y=data_y[0:1320*16],data_y[1320*16:26399+1]
 
from numpy import nan
from numpy import isnan
from pandas import read_csv
from pandas import to_numeric
 
from sklearn.metrics import r2_score 
import lightgbm as lgb
# multivariate multi-step encoder-decoder lstm
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
 
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from numpy.random import seed 
import numpy as np
import xgboost as xgb
import pandas as pd
#from sklearn.metrics import roc_auc_score
from sklearn.metrics import explained_variance_score
import matplotlib.pyplot as plt
 
from hyperopt import STATUS_OK,STATUS_RUNNING, fmin, hp, tpe,space_eval, partial
 
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
 
 
 
##训练参数
SEED = 314159265
VALID_SIZE = 0.25
def model_run(params):
    
    
    print("Training with params: ")
    print(params)
    # train
    print("Training with params: ")
    print(params)
    print("training...")
    model_lgb = lgb.train(params, train_data, num_boost_round=1000, valid_sets=[test_data],early_stopping_rounds=30)
    print("Validating...")
    # predict
    check =model_lgb.predict(test_x)
    print("explained_variance_score...")
    score = get_score(test_y, check)
    print("pr...")
    print('The mse of prediction is: {:.6f}'.format(score))
    
   ## print("Predict test set...")
   ## test_prediction = gbm.predict(xgb.DMatrix(test[features]), ntree_limit=gbm.best_iteration+1)
    return {
        'loss': score,
        'status': STATUS_OK,
        'stats_running': STATUS_RUNNING
    }
 
def optimize(
             #trials, 
             random_state=SEED):
    
   # 自定义hyperopt的参数空间    
    space = {"n_iter":hp.choice("n_iter",range(50,200)), 
         "eta":hp.quniform("eta",0.05,0.5,0.05),
         'eval_metric': 'rmse',
         'objective': 'regression',
         'boosting_type': 'gbdt',
         'learning_rate': hp.quniform("learning_rate",0.05,0.3,0.02),
         'num_leaves': 6,
         'max_depth': 4,
         'min_child_weight': hp.quniform("min_child_weight",2,10,1)
         }
    
    print("---------开始训练参数----------")
   # best = fmin(model_run, space, algo=tpe.suggest, max_evals=1)
    print("------------partial-------------")
    ##获取最优的参数
    algo = partial(tpe.suggest, n_startup_jobs=1)
    print("----------fmin---------------")
    best = fmin(model_run, space, algo=algo, max_evals=1000, pass_expr_memo_ctrl=None)
    print("-------------------------")
    best_params = space_eval(space, best)
    print("BEST PARAMETERS: " + str(best_params))
    return best_params
 
##定义计分函数
def get_score(pre,real):
    temp=[]
    pre_t=[]
    real_t=[]
    pre=pre.round().astype(int)
    
    for i in range(60):
        for j in range(4):
            pre_t.append(pre[1320*j+22*i:1320*j+22*(i+1)])
            real_t.append(real[1320*j+22*i:1320*j+22*(i+1)])
        temp.append(((mean_squared_error(pre_t,real_t))**0.5)/np.mean(real_t))
    return sum(temp)/60
print("---------DMatrix----------")
print(train_x, train_y)
print(test_x, test_y)
exit(0)
train_data = lgb.Dataset(data=train_x,label=train_y)
test_data = lgb.Dataset(data=test_x,label=test_y)
print("---------开始优化参数----------")
best_params=optimize()
#print(test_prediction)
print("---------优化完成----------")
print(best_params)
 
##训练模型
 
print("---------正式训练模型----------")
model_lgb = lgb.train(best_params, train_data, num_boost_round=300, valid_sets=[test_data],early_stopping_rounds=30)
print("---------正式预测模型----------")
print("Predict test set...")
test_prediction = model_lgb.predict(data[1320*20:1320*24])
test_prediction1=model_lgb.predict(test_x)
print("---------预测完成----------")
print(best_params)
print(test_prediction.shape)
test_prediction=test_prediction.round().astype(int)
f = open('./car_re_lgb.txt', 'w')
total = 0
for id in range(1320*4):
    str1 =str(test_prediction[total])
    str1 += '\n'
    total += 1
    f.write(str1)
f.close()
print("持久化完成")
test_prediction1=test_prediction1.round().astype(int)
score =get_score(test_y, test_prediction1)
print(1-score)
# ————————————————
# 版权声明：本文为CSDN博主「喝粥也会胖的唐僧」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/zhou_438/article/details/101351557