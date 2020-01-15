# coding:utf-8
# 导入warnings包，利用过滤器来实现忽略警告语句。
import warnings

warnings.filterwarnings('ignore')

# GBDT
from sklearn.ensemble import GradientBoostingRegressor
# XGBoost
import xgboost as xgb
# LightGBM
import lightgbm as lgb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import pickle
import multiprocessing
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC, LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

train_path = '/Users/gongbin/Documents/数据竞赛/Datawhale/team-learning/数据竞赛（房租预测）/Data/train_data.csv'
test_path = '/Users/gongbin/Documents/数据竞赛/Datawhale/team-learning/数据竞赛（房租预测）/Data/test_a.csv'
data_train_origin = pd.read_csv(train_path)
data_train = data_train_origin.copy()
data_train['Type'] = 'Train'
data_test_origin = pd.read_csv(test_path)
data_test = data_test_origin.copy()
data_test['Type'] = 'Test'
data_all = pd.concat([data_train, data_test], ignore_index=True)


def preprocessingData(data):
    # 填充缺失值
    data['rentType'][data['rentType'] == '--'] = '未知方式'  # 进行填充

    # 转换object类型数据
    columns = ['rentType', 'communityName', 'houseType', 'houseFloor', 'houseToward', 'houseDecoration', 'region',
               'plate']

    for feature in columns:
        data[feature] = LabelEncoder().fit_transform(data[feature])

    # 将buildYear列转换为整型数据
    buildYearmean = pd.DataFrame(data[data['buildYear'] != '暂无信息']['buildYear'].mode())  # 用平均数填充
    data.loc[data[data['buildYear'] == '暂无信息'].index, 'buildYear'] = buildYearmean.iloc[0, 0]
    data['buildYear'] = data['buildYear'].astype('int')  # 年份转成整数

    # 处理pv和uv的空值
    data['pv'].fillna(data['pv'].mean(), inplace=True)
    data['uv'].fillna(data['uv'].mean(), inplace=True)
    data['pv'] = data['pv'].astype('int')
    data['uv'] = data['uv'].astype('int')

    # 分割交易时间
    def month(x):
        month = int(x.split('/')[1])
        return month

    def year(x):
        year = int(x.split('/')[0])
        return year

    data['month'] = data['tradeTime'].apply(lambda x: month(x))
    data['year'] = data['tradeTime'].apply(lambda x: year(x))

    # 去掉部分特征
    data.drop('city', axis=1, inplace=True)
    data.drop('tradeTime', axis=1, inplace=True)
    data.drop('ID', axis=1, inplace=True)
    return data


data_train = preprocessingData(data_train)


def IF_drop(train):
    IForest = IsolationForest(contamination=0.01)
    IForest.fit(train["tradeMoney"].values.reshape(-1, 1))
    y_pred = IForest.predict(train["tradeMoney"].values.reshape(-1, 1))
    drop_index = train.loc[y_pred == -1].index
    print(drop_index)
    train.drop(drop_index, inplace=True)
    return train


data_train = IF_drop(data_train)


def dropData(train):
    # 丢弃部分异常值
    train = train[train.area <= 200]
    train = train[(train.tradeMoney <= 16000) & (train.tradeMoney >= 700)]
    train.drop(train[(train['totalFloor'] == 0)].index, inplace=True)
    return train


# 数据集异常值处理
data_train = dropData(data_train)


def newfeature(data):
    # 将houseType转为'Room'，'Hall'，'Bath'    用于判断单个房间数量对结果的影响
    '''
    def Room(x):
        Room = int(x.split('室')[0])
        return Room
    def Hall(x):
        Hall = int(x.split("室")[1].split("厅")[0])
        return Hall
    def Bath(x):
        Bath = int(x.split("室")[1].split("厅")[1].split("卫")[0])
        return Bath

    data['Room'] = data['houseType'].apply(lambda x: Room(x))
    data['Hall'] = data['houseType'].apply(lambda x: Hall(x))
    data['Bath'] = data['houseType'].apply(lambda x: Bath(x))
    data['Room_Bath'] = (data['Bath']+1) / (data['Room']+1)
    # 填充租房类型    根据
    '''

    # data['day'] = data['tradeTime'].apply(lambda x: day(x))# 结果变差
    #     data['pv/uv'] = data['pv'] / data['uv']
    #     data['房间总数'] = data['室'] + data['厅'] + data['卫']

    # 合并部分配套设施特征
    data['trainsportNum'] = 5 * data['subwayStationNum'] / data['subwayStationNum'].mean() + data['busStationNum'] / \
                            data[
                                'busStationNum'].mean()
    data['all_SchoolNum'] = 2 * data['interSchoolNum'] / data['interSchoolNum'].mean() + data['schoolNum'] / data[
        'schoolNum'].mean() \
                            + data['privateSchoolNum'] / data['privateSchoolNum'].mean()
    data['all_hospitalNum'] = 2 * data['hospitalNum'] / data['hospitalNum'].mean() + \
                              data['drugStoreNum'] / data['drugStoreNum'].mean()
    data['all_mall'] = data['mallNum'] / data['mallNum'].mean() + \
                       data['superMarketNum'] / data['superMarketNum'].mean()
    data['otherNum'] = data['gymNum'] / data['gymNum'].mean() + data['bankNum'] / data['bankNum'].mean() + \
                       data['shopNum'] / data['shopNum'].mean() + 2 * data['parkNum'] / data['parkNum'].mean()

    data.drop(['subwayStationNum', 'busStationNum',
               'interSchoolNum', 'schoolNum', 'privateSchoolNum',
               'hospitalNum', 'drugStoreNum', 'mallNum', 'superMarketNum', 'gymNum', 'bankNum', 'shopNum', 'parkNum'],
              axis=1, inplace=True)
    # 提升0.0005

    #     data['houseType_1sumcsu']=data['Bath'].map(lambda x:str(x))+data['month'].map(lambda x:str(x))
    #     data['houseType_2sumcsu']=data['Bath'].map(lambda x:str(x))+data['communityName']
    #     data['houseType_3sumcsu']=data['Bath'].map(lambda x:str(x))+data['plate']

    # data.drop('houseType', axis=1, inplace=True)
    # data.drop('tradeTime', axis=1, inplace=True)

    data["area"] = data["area"].astype(int)

    # categorical_feats = ['rentType', 'houseFloor', 'houseToward', 'houseDecoration', 'communityName','region', 'plate']
    categorical_feats = ['rentType', 'houseFloor', 'houseToward', 'houseDecoration', 'region', 'plate', 'cluster']

    return data, categorical_feats


data_train, categorical_feats = newfeature(data_train)

data_train.to_csv('数据集/ProccessedData.csv')
