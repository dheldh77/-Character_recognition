import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras
import sklearn

plt.style.use('seaborn')
sns.set(font_scale=2.5)
import missingno as msno

#ignore warnings
import warnings
warnings.filterwarnings('ignore') 

df_train = pd.read_csv('train.csv', encoding='euc-kr')

print(df_train.shape)
print(df_train.columns)

# 혈액형 별 치매로부터 안전할 확률
print('blood_type vs pass_or_fail')
print(df_train[['blood_type', 'pass_or_fail']].groupby(['blood_type'], as_index=True).mean())

# 성별 치매로부터 안전할 확률
print('gender vs pass_or_fail')
print(df_train[['gender', 'pass_or_fail']].groupby(['gender'], as_index=True).mean())

heatmap_data = df_train[['pass_or_fail', 'gender', 'age', 'blood_type', 'height', 'weight', 'past_diagnostic_record']]
colormap = plt.cm.RdBu
plt.figure(figsize=(14, 12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(heatmap_data.astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True, annot_kws={"size": 16})

plt.show()
del heatmap_data
