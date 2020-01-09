import numpy as np
from collections import Counter
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
data_train = pd.read_csv('数据集/train_data.csv')
data_train['Type'] = 'Train'
data_test = pd.read_csv('数据集/test_a.csv')
data_test['Type'] = 'Test'
data_all = pd.concat([data_train, data_test], ignore_index=True)

data = data_train


# print(data_train.rentType.value_counts())
# print(data_train[data_train['buildYear'] != '暂无信息']['buildYear'])
# print(data_train[data_train['buildYear'] != '暂无信息']['buildYear'])

def preprocessingData(data):
    # 缺失值处理
    data.rentType[data.rentType == '--'] = '未知方式'  # 从task1可知，‘--’和‘未知方式’含义一样，故统一
    columns = ['rentType', 'communityName', 'houseType', 'houseFloor', 'houseToward', 'houseDecoration', 'region',
               'plate']
    for feature in columns:
        data[feature] = LabelEncoder().fit_transform(data[feature])  # 利用sklearn的LabelEncoder类
    # print(data_train.rentType)

    buildYearmode = pd.DataFrame(
        data[data['buildYear'] != '暂无信息']['buildYear'].mode())  # 获取buildYear的众数，因为众数可以不止一个，所以返回的是dataframe
    data.loc[data[data['buildYear'] == '暂无信息'].index, 'buildYear'] = buildYearmode.iloc[0, 0]  # 用众数填充缺失值
    data['buildYear'] = data['buildYear'].astype('int')  # 将buildYear列转换为整型数据

    # 分割交易时间，字符分割加上强转
    def month(x):
        month = int(x.split('/')[1])
        return month

    def day(x):
        day = int(x.split('/')[2])
        return day

    data['month'] = data['tradeTime'].apply(lambda x: month(x))
    data['day'] = data['tradeTime'].apply(lambda x: day(x))
    # 去掉部分特征
    data.drop('city', axis=1, inplace=True)
    data.drop('tradeTime', axis=1, inplace=True)
    data.drop('ID', axis=1, inplace=True)
    return data


# data_train = preprocessingData(data_train)
# print(data_train.info())

# buildYearmean = pd.DataFrame(data[data['buildYear'] != '暂无信息']['buildYear'].mode())
# print(buildYearmean)
# print(data['buildYear'].value_counts())
# data.loc[data[data['buildYear'] == '暂无信息'].index, 'buildYear'] = buildYearmean.iloc[0, 0]
# print(data['buildYear'].value_counts())

# clean data, 孤立森林去异常还是很赞的
def IF_drop(train):
    IForest = IsolationForest(contamination=0.01)
    IForest.fit(train["tradeMoney"].values.reshape(-1, 1))
    y_pred = IForest.predict(train["tradeMoney"].values.reshape(-1, 1))
    drop_index = train.loc[y_pred == -1].index
    print(drop_index)
    train.drop(drop_index, inplace=True)
    return train


data_train = IF_drop(data_train)
print(data_train)


def detect_outliers(df, n, features):  # 一种箱型图去异常的方法
    outlier_indices = []
    for col in features:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)
    return multiple_outliers

def dropData(train):
    # 丢弃部分异常值
    train = train[train.area <= 200]
    train = train[(train.tradeMoney <= 16000) & (train.tradeMoney >= 700)]
    train.drop(train[(train['totalFloor'] == 0)].index, inplace=True)
    return train


# 数据集异常值处理
data_train = dropData(data_train)

# 处理异常值后再次查看面积和租金分布图
plt.figure(figsize=(15, 5))
sns.boxplot(data_train.area)
plt.savefig('homework_day2/area.png')
plt.show()
plt.figure(figsize=(15, 5))
sns.boxplot(data_train.tradeMoney)
plt.savefig('homework_day2/tradeMoney.png')
plt.show()


def cleanData(data):  # 这里有我踩过的坑，只能用&不能用and
    data.drop(data[(data['region'] == 'RG00001') & (data['tradeMoney'] < 1000) & (data['area'] > 50)].index,
              inplace=True)
    data.drop(data[(data['region'] == 'RG00001') & (data['tradeMoney'] > 25000)].index, inplace=True)
    data.drop(data[(data['region'] == 'RG00001') & (data['area'] > 250) & (data['tradeMoney'] < 20000)].index,
              inplace=True)
    data.drop(data[(data['region'] == 'RG00001') & (data['area'] > 400) & (data['tradeMoney'] > 50000)].index,
              inplace=True)
    data.drop(data[(data['region'] == 'RG00001') & (data['area'] > 100) & (data['tradeMoney'] < 2000)].index,
              inplace=True)
    data.drop(data[(data['region'] == 'RG00002') & (data['area'] < 100) & (data['tradeMoney'] > 60000)].index,
              inplace=True)
    data.drop(data[(data['region'] == 'RG00003') & (data['area'] < 300) & (data['tradeMoney'] > 30000)].index,
              inplace=True)
    data.drop(data[(data['region'] == 'RG00003') & (data['tradeMoney'] < 500) & (data['area'] < 50)].index,
              inplace=True)
    data.drop(data[(data['region'] == 'RG00003') & (data['tradeMoney'] < 1500) & (data['area'] > 100)].index,
              inplace=True)
    data.drop(data[(data['region'] == 'RG00003') & (data['tradeMoney'] < 2000) & (data['area'] > 300)].index,
              inplace=True)
    data.drop(data[(data['region'] == 'RG00003') & (data['tradeMoney'] > 5000) & (data['area'] < 20)].index,
              inplace=True)
    data.drop(data[(data['region'] == 'RG00003') & (data['area'] > 600) & (data['tradeMoney'] > 40000)].index,
              inplace=True)
    data.drop(data[(data['region'] == 'RG00004') & (data['tradeMoney'] < 1000) & (data['area'] > 80)].index,
              inplace=True)
    data.drop(data[(data['region'] == 'RG00006') & (data['tradeMoney'] < 200)].index, inplace=True)
    data.drop(data[(data['region'] == 'RG00005') & (data['tradeMoney'] < 2000) & (data['area'] > 180)].index,
              inplace=True)
    data.drop(data[(data['region'] == 'RG00005') & (data['tradeMoney'] > 50000) & (data['area'] < 200)].index,
              inplace=True)
    data.drop(data[(data['region'] == 'RG00006') & (data['area'] > 200) & (data['tradeMoney'] < 2000)].index,
              inplace=True)
    data.drop(data[(data['region'] == 'RG00007') & (data['area'] > 100) & (data['tradeMoney'] < 2500)].index,
              inplace=True)
    data.drop(data[(data['region'] == 'RG00010') & (data['area'] > 200) & (data['tradeMoney'] > 25000)].index,
              inplace=True)
    data.drop(data[(data['region'] == 'RG00010') & (data['area'] > 400) & (data['tradeMoney'] < 15000)].index,
              inplace=True)
    data.drop(data[(data['region'] == 'RG00010') & (data['tradeMoney'] < 3000) & (data['area'] > 200)].index,
              inplace=True)
    data.drop(data[(data['region'] == 'RG00010') & (data['tradeMoney'] > 7000) & (data['area'] < 75)].index,
              inplace=True)
    data.drop(data[(data['region'] == 'RG00010') & (data['tradeMoney'] > 12500) & (data['area'] < 100)].index,
              inplace=True)
    data.drop(data[(data['region'] == 'RG00004') & (data['area'] > 400) & (data['tradeMoney'] > 20000)].index,
              inplace=True)
    data.drop(data[(data['region'] == 'RG00008') & (data['tradeMoney'] < 2000) & (data['area'] > 80)].index,
              inplace=True)
    data.drop(data[(data['region'] == 'RG00009') & (data['tradeMoney'] > 40000)].index, inplace=True)
    data.drop(data[(data['region'] == 'RG00009') & (data['area'] > 300)].index, inplace=True)
    data.drop(data[(data['region'] == 'RG00009') & (data['area'] > 100) & (data['tradeMoney'] < 2000)].index,
              inplace=True)
    data.drop(data[(data['region'] == 'RG00011') & (data['tradeMoney'] < 10000) & (data['area'] > 390)].index,
              inplace=True)
    data.drop(data[(data['region'] == 'RG00012') & (data['area'] > 120) & (data['tradeMoney'] < 5000)].index,
              inplace=True)
    data.drop(data[(data['region'] == 'RG00013') & (data['area'] < 100) & (data['tradeMoney'] > 40000)].index,
              inplace=True)
    data.drop(data[(data['region'] == 'RG00013') & (data['area'] > 400) & (data['tradeMoney'] > 50000)].index,
              inplace=True)
    data.drop(data[(data['region'] == 'RG00013') & (data['area'] > 80) & (data['tradeMoney'] < 2000)].index,
              inplace=True)
    data.drop(data[(data['region'] == 'RG00014') & (data['area'] > 300) & (data['tradeMoney'] > 40000)].index,
              inplace=True)
    data.drop(data[(data['region'] == 'RG00014') & (data['tradeMoney'] < 1300) & (data['area'] > 80)].index,
              inplace=True)
    data.drop(data[(data['region'] == 'RG00014') & (data['tradeMoney'] < 8000) & (data['area'] > 200)].index,
              inplace=True)
    data.drop(data[(data['region'] == 'RG00014') & (data['tradeMoney'] < 1000) & (data['area'] > 20)].index,
              inplace=True)
    data.drop(data[(data['region'] == 'RG00014') & (data['tradeMoney'] > 25000) & (data['area'] > 200)].index,
              inplace=True)
    data.drop(data[(data['region'] == 'RG00014') & (data['tradeMoney'] < 20000) & (data['area'] > 250)].index,
              inplace=True)
    data.drop(data[(data['region'] == 'RG00005') & (data['tradeMoney'] > 30000) & (data['area'] < 100)].index,
              inplace=True)
    data.drop(data[(data['region'] == 'RG00005') & (data['tradeMoney'] < 50000) & (data['area'] > 600)].index,
              inplace=True)
    data.drop(data[(data['region'] == 'RG00005') & (data['tradeMoney'] > 50000) & (data['area'] > 350)].index,
              inplace=True)
    data.drop(data[(data['region'] == 'RG00006') & (data['tradeMoney'] > 4000) & (data['area'] < 100)].index,
              inplace=True)
    data.drop(data[(data['region'] == 'RG00006') & (data['tradeMoney'] < 600) & (data['area'] > 100)].index,
              inplace=True)
    data.drop(data[(data['region'] == 'RG00006') & (data['area'] > 165)].index, inplace=True)
    data.drop(data[(data['region'] == 'RG00012') & (data['tradeMoney'] < 800) & (data['area'] < 30)].index,
              inplace=True)
    data.drop(data[(data['region'] == 'RG00007') & (data['tradeMoney'] < 1100) & (data['area'] > 50)].index,
              inplace=True)
    data.drop(data[(data['region'] == 'RG00004') & (data['tradeMoney'] > 8000) & (data['area'] < 80)].index,
              inplace=True)
    data.loc[(data['region'] == 'RG00002') & (data['area'] > 50) & (data['rentType'] == '合租'), 'rentType'] = '整租'
    data.loc[(data['region'] == 'RG00014') & (data['rentType'] == '合租') & (data['area'] > 60), 'rentType'] = '整租'
    data.drop(data[(data['region'] == 'RG00008') & (data['tradeMoney'] > 15000) & (data['area'] < 110)].index,
              inplace=True)
    data.drop(data[(data['region'] == 'RG00008') & (data['tradeMoney'] > 20000) & (data['area'] > 110)].index,
              inplace=True)
    data.drop(data[(data['region'] == 'RG00008') & (data['tradeMoney'] < 1500) & (data['area'] < 50)].index,
              inplace=True)
    data.drop(data[(data['region'] == 'RG00008') & (data['rentType'] == '合租') & (data['area'] > 50)].index,
              inplace=True)
    data.drop(data[(data['region'] == 'RG00015')].index, inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data


data_train = cleanData(data_train)
print(data_train)
