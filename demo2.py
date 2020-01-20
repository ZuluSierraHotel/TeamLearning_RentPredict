import pandas as pd

feature = pd.read_csv('数据集/ProccessedTrainData.csv', index_col=0)
test = pd.read_csv('数据集/ProccessedTestData.csv', index_col=0)

print(feature.info)
print(test.info)