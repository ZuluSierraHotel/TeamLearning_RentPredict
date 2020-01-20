import lightgbm as lgb
import sklearn
import numpy
import hyperopt
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import colorama
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.metrics import r2_score

folds = KFold(n_splits=5, shuffle=True, random_state=0)

test = pd.read_csv('数据集/ProccessedTestData.csv', index_col=0)
# test.pop('Unnamed: 0')
test.pop('tradeMoney')

categorical_feats = ['rentType', 'houseFloor', 'houseToward', 'houseDecoration',  'region', 'plate','cluster']
feature = pd.read_csv('数据集/ProccessedTrainData.csv', index_col=0)
label = feature.pop('tradeMoney')
# feature.pop('Unnamed: 0')




# 1
y_pre_list = []
r2_list = []
train_feat = pd.Series()
for fold_, (trn_idx, val_idx) in enumerate(folds.split(feature.values, label)):
    print("fold {}".format(fold_))
    trn_data = lgb.Dataset(feature.iloc[trn_idx], label = label[trn_idx], categorical_feature=categorical_feats)
    val_data = lgb.Dataset(feature.iloc[val_idx], label = label[val_idx], categorical_feature=categorical_feats)

    num_round = 10000
    clf = lgb.train(params, trn_data, num_round,valid_sets=[trn_data, val_data], verbose_eval=500,
                    early_stopping_rounds=200)
    y_pre = clf.predict(feature.iloc[val_idx], num_iteration=clf.best_iteration)
    r2 = r2_score(y_pre,label[val_idx])
    r2_list.append(r2)
    train_feat = train_feat.append(pd.Series(y_pre,index=val_idx))
    y_pre_test = clf.predict(test,num_iteration=clf.best_iteration)
    y_pre_list.append(y_pre_test)
print('r2 score{:}'.format(r2))
print('r2:{:}'.format(np.mean(r2_list)))

y_pred_final = (y_pre_list[0]+y_pre_list[1]+y_pre_list[2]+y_pre_list[3]+y_pre_list[4])/5
feature['pre'] = train_feat
test['pre'] = y_pred_final