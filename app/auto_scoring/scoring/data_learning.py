import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import math

warnings.filterwarnings("ignore")
from subprocess import check_output
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pickle
import joblib
from sklearn.linear_model import LinearRegression

# 환자들에 대한 정보를 선형 모델로 기계학습 및 인자로 전달된 정보로 임상치매척도 판단
def learning_about_data():
    # train 데이터 읽어옴
    data = pd.read_csv('./scoring/train.csv',encoding='euc-kr')
    # 필요없는 속성 제거
    data.drop(["name", "Unnamed: 0"],axis=1, inplace = True)
    print(data);
    # CDR을 one-hot encoding 해줌
    # data = pd.get_dummies(data, columns=['CDR'], prefix='CDR')

    # y_0 = data['CDR_0.0'].values
    # y_1 = data['CDR_0.5'].values
    # y_2 = data['CDR_1.0'].values
    # y_3 = data['CDR_2.0'].values

    # x_data = data.drop(['CDR_0.0','CDR_0.5','CDR_1.0','CDR_2.0'],axis = 1)
    # y_list = [y_0, y_1, y_2, y_3]

    # 예측값 제거
    y = data['CDR'].values
    x_data = data.drop(['CDR'],axis = 1)

    # 평균값
    avg = {
        "ASF": data['ASF'].mean(),
        "nWBV" : data['nWBV'].mean(),
        "eTIV" : round(data['eTIV'].mean()),
        "MMSE" : data['MMSE'].mean(),
        "SES" : round(data['SES'].mean()),
        "MR_delay" : round(data['MR_delay'].mean()),
    }

    # 머신러닝
    model = LinearRegression()
    model.fit(x_data, y)

    # 저장
    joblib.dump(model, 'data_model.pkl')

    return avg


def check_CDR(check_object):
    #불러오기
    load_model = joblib.load('data_model.pkl')

    object = {
            "gender" : [check_object['gender']],
            "age" : [check_object['age']],
            "ASF": [check_object['ASF']],
            "nWBV" : [check_object['nWBV']],
            "eTIV" : [check_object['eTIV']],
            "CDR" : [0],
            "MMSE" : [check_object['MMSE']],
            "SES" : [check_object['SES']],
            "educ" : [check_object['educ']],
            "hand" : [check_object['hand']],
            "MR_delay" : [check_object["MR_delay"]],
        }

    df = pd.DataFrame(object)
    temp = df.drop('CDR', axis = 1)


    prediction = load_model.predict(temp)

    if prediction < 0.1 :
        prediction = 0.0
    elif prediction >= 0.1 and prediction < 0.5:
        prediction = 0.5
    elif prediction >=0.5 and prediction < 1.0:
        prediction = 1.0
    else:
        prediction = 2.0
    return prediction
