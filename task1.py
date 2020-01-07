import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

'''
题目类型：回归问题
评价指标：R**2. R**2作为拟合优度的统计量，其取值在0-1之间，越大表示拟合优度越好；
         评价指标应该是会对模型以及损失函数的选择有影响吧，这一点在调包的时候再仔细考虑
'''

data_train = pd.read_csv('数据集/train_data.csv')
data_test = pd.read_csv('数据集/test_a.csv')
# print(data_train.info(), data_train.describe())
# print(data_test.info(), data_test.describe())
data_train['Type'] = 'Train'
data_test['Type'] = 'Test'
data_all = pd.concat([data_train, data_test], ignore_index=True)  # 合并数据
'''
观察数据：
训练集：41440*51，特征中9项是实数，30项是整数，11项非数值型，需要数值化。标签是浮点型，进一步验证是回归问题。
        存在缺失值。
测试集：2469*50，特征和训练集一致，也存在缺失值。
'''


def CreateDictionary():
    '''
    创建一个特征字段和字段解释的字典，以便分析特征时查询
    :return:
    '''
    with open('数据集/字段.txt', 'r') as f:
        strings = f.readlines()
    keys = []
    explanations = []
    for s in strings:
        key = s.split('——')[0].replace(' ', '').replace('\r', '').replace('\n', '')
        explanation = s.split('——')[1].replace(' ', '').replace('\r', '').replace('\n', '').replace('\u3000', '')
        keys.append(key)
        explanations.append(explanation)
    dictionary = dict(zip(keys, explanations))
    return dictionary


CreateDictionary()

categorical_feas = [col for col in data_test.columns if str(data_test[col].dtype) == 'object']
numerical_feas = [col for col in data_test.columns if str(data_test[col].dtype) != 'object']
# print(categorical_feas)
# print(numerical_feas)
'''
区分数值型特征和类别型特征，方便后续的统计分析和数值化
'''


# 缺失值分析
def missing_values(df):
    '''
    借鉴范例，创建一个缺失值统计表
    :param df:
    :return:
    '''
    alldata_na = pd.DataFrame(df.isnull().sum(), columns={'missingNum'})
    alldata_na['existNum'] = len(df) - alldata_na['missingNum']
    alldata_na['sum'] = len(df)
    alldata_na['missingRatio'] = alldata_na['missingNum'] / len(df) * 100
    alldata_na['dtype'] = df.dtypes
    # ascending：默认True升序排列；False降序排列
    alldata_na = alldata_na[alldata_na['missingNum'] > 0].reset_index().sort_values(by=['missingNum', 'index'],
                                                                                    ascending=[False, True])
    alldata_na.set_index('index', inplace=True)
    return alldata_na


alldata_na = missing_values(data_train)
# print(alldata_na)
'''
缺失值分析，pv（该板块当月租客浏览网页次数),uv(该板块当月租客浏览网页总人数)字段存在缺失，
拟打算根据和label的相关性大小，决定是否填充，如何填充
'''


# 单调性分析
def incresing(vals):
    cnt = 0
    len_ = len(vals)
    for i in range(len_ - 1):
        if vals[i + 1] > vals[i]:
            cnt += 1
    return cnt


fea_cols = [col for col in data_train.columns]
for col in fea_cols:
    cnt = incresing(data_train[col].values)
    if cnt / data_train.shape[0] >= 0.55:
        print('单调特征：', col)
        print('单调特征值个数：', cnt)
        print('单调特征值比例：', cnt / data_train.shape[0])
'''
单调性分析：其中的参数0.55让我觉得有点好奇，是一个经验值吗？
数据顺序可以调换，所以单调性分析感觉意义不大吧
'''

# 特征nunique分布
for feature in categorical_feas:
    print(feature + "的特征分布如下：")
    print(data_train[feature].value_counts())
    if feature != 'communityName':  # communityName值太多，暂且不看图表
        plt.hist(data_all[feature].value_counts(), bins=3)
        plt.savefig('homework_day1/' + feature + '.png')
        plt.show()

for feature in numerical_feas:
    print(feature + "的特征分布如下：")
    print(data_train[feature].value_counts())
    plt.hist(data_all[feature], bins=3)
    plt.savefig('homework_day1/' + feature + '.png')
    plt.show()

'''
粗略看了下所有特征的分布
'''

# 统计特征值出现频次大于100的特征
for feature in categorical_feas:
    df_value_counts = pd.DataFrame(data_train[feature].value_counts())
    df_value_counts = df_value_counts.reset_index()
    df_value_counts.columns = [feature, 'counts']  # change column names
    print(df_value_counts[df_value_counts['counts'] >= 100])

# Labe 分布
print(data_train['tradeMoney'].describe())
fig, axes = plt.subplots(2, 3, figsize=(20, 5))
fig.set_size_inches(20, 12)
sns.distplot(data_train['tradeMoney'], ax=axes[0][0])
sns.distplot(data_train[(data_train['tradeMoney'] <= 20000)]['tradeMoney'], ax=axes[0][1])
sns.distplot(data_train[(data_train['tradeMoney'] > 20000) & (data_train['tradeMoney'] <= 50000)]['tradeMoney'],
             ax=axes[0][2])
sns.distplot(data_train[(data_train['tradeMoney'] > 50000) & (data_train['tradeMoney'] <= 100000)]['tradeMoney'],
             ax=axes[1][0])
sns.distplot(data_train[(data_train['tradeMoney'] > 100000)]['tradeMoney'], ax=axes[1][1])
plt.savefig('homework_day1/label_distribution.png')
plt.show()
print("money<=10000", len(data_train[(data_train['tradeMoney'] <= 10000)]['tradeMoney']))
print("10000<money<=20000",
      len(data_train[(data_train['tradeMoney'] > 10000) & (data_train['tradeMoney'] <= 20000)]['tradeMoney']))
print("20000<money<=50000",
      len(data_train[(data_train['tradeMoney'] > 20000) & (data_train['tradeMoney'] <= 50000)]['tradeMoney']))
print("50000<money<=100000",
      len(data_train[(data_train['tradeMoney'] > 50000) & (data_train['tradeMoney'] <= 100000)]['tradeMoney']))
print("100000<money", len(data_train[(data_train['tradeMoney'] > 100000)]['tradeMoney']))
'''
发现label的数据不均衡,极差，方差都很大，存在0值。而且主要分布在小于10000的区间内，分布也不是正态的，有峰值，有偏置。
'''

# 相关性分析
corrmat = data_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
plt.savefig('homework_day1/correlation_heatmap.png')
plt.show()
'''
绘制相关系数的热制图，观察特征与特征，特征与标签之间的相关关系。
发现有些特征之间相关关系明显，但是和标签的相关性头不是太大，感觉有点奇怪，初步怀疑是标签选择的不好，因为标签的分布本身不太均衡
'''

plt.figure()
k = 10
cols = corrmat.nlargest(k, 'tradeMoney')['tradeMoney'].index
cm = np.corrcoef(data_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,
                 xticklabels=cols.values)
plt.savefig('homework_day1/correlation_value_figure.png')
plt.show()
'''
绘制包含具体相关系数值的热制图，观察最影响tradeMoney的九个特征，发现相关系关系的确不大。
'''
