import pandas as pd
import warnings

warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder

# 读取数据
train = pd.read_csv('数据集/train_data.csv')
test = pd.read_csv('数据集/test_a.csv')
test_label = pd.read_csv('数据集/评分文件/sub_a_913.csv')
test['tradeMoney'] = test_label

target_train = train.pop('tradeMoney')
target_test = test.pop('tradeMoney')

# 相关系数法特征选择
from sklearn.feature_selection import SelectKBest

print(train.shape)

sk = SelectKBest(k=150)
new_train = sk.fit_transform(train, target_train)
print(new_train.shape)

# 获取对应列索引
select_columns = sk.get_support(indices=True)
# print(select_columns)

# 获取对应列名
# print(test.columns[select_columns])
select_columns_name = test.columns[select_columns]
new_test = test[select_columns_name]
print(new_test.shape)
# Lasso回归
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.1)
lasso.fit(new_train, target_train)
# 预测测试集和训练集结果
y_pred_train = lasso.predict(new_train)

y_pred_test = lasso.predict(new_test)

# 对比结果
from sklearn.metrics import r2_score

score_train = r2_score(y_pred_train, target_train)
print("训练集结果：", score_train)
score_test = r2_score(y_pred_test, target_test)
print("测试集结果：", score_test)

# Wrapper

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
rfe = RFE(lr, n_features_to_select=160)
rfe.fit(train, target_train)

RFE(estimator=LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
                               normalize=False),
    n_features_to_select=40, step=1, verbose=0)

select_columns = [f for f, s in zip(train.columns, rfe.support_) if s]
print(select_columns)
new_train = train[select_columns]
new_test = test[select_columns]

# Lasso回归
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.1)
lasso.fit(new_train, target_train)
# 预测测试集和训练集结果
y_pred_train = lasso.predict(new_train)

y_pred_test = lasso.predict(new_test)

# 对比结果
from sklearn.metrics import r2_score

score_train = r2_score(y_pred_train, target_train)
print("训练集结果：", score_train)
score_test = r2_score(y_pred_test, target_test)
print("测试集结果：", score_test)

# Embedded
# 基于惩罚项的特征选择法
# Lasso(l1)和Ridge(l2)

from sklearn.linear_model import Ridge

ridge = Ridge(alpha=5)
ridge.fit(train, target_train)

Ridge(alpha=5, copy_X=True, fit_intercept=True, max_iter=None, normalize=False,
      random_state=None, solver='auto', tol=0.001)

# 特征系数排序
coefSort = ridge.coef_.argsort()
print(coefSort)

# 特征系数
featureCoefSore = ridge.coef_[coefSort]
print(featureCoefSore)

select_columns = [f for f, s in zip(train.columns, featureCoefSore) if abs(s) > 0.0000005]
# 选择绝对值大于0.0000005的特征

new_train = train[select_columns]
new_test = test[select_columns]
# Lasso回归
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.1)
lasso.fit(new_train, target_train)
# 预测测试集和训练集结果
y_pred_train = lasso.predict(new_train)

y_pred_test = lasso.predict(new_test)

# 对比结果
from sklearn.metrics import r2_score

score_train = r2_score(y_pred_train, target_train)
print("训练集结果：", score_train)
score_test = r2_score(y_pred_test, target_test)
print("测试集结果：", score_test)

# Embedded
# 基于树模型的特征选择法
# 随机森林 平均不纯度减少（mean decrease impurity


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
# 训练随机森林模型，并通过feature_importances_属性获取每个特征的重要性分数。rf = RandomForestRegressor()
rf.fit(train, target_train)
print("Features sorted by their score:")
print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), train.columns),
             reverse=True))

select_columns = [f for f, s in zip(train.columns, rf.feature_importances_) if abs(s) > 0.00005]
# 选择绝对值大于0.00005的特征

new_train = train[select_columns]
new_test = test[select_columns]

# Lasso回归
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.1)
lasso.fit(new_train, target_train)
# 预测测试集和训练集结果
y_pred_train = lasso.predict(new_train)

y_pred_test = lasso.predict(new_test)

# 对比结果
from sklearn.metrics import r2_score

score_train = r2_score(y_pred_train, target_train)
print("训练集结果：", score_train)
score_test = r2_score(y_pred_test, target_test)
print("测试集结果：", score_test)
