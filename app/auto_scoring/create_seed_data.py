from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import pandas as pd
import random
import csv
import json

# 이름 긁어오기
def get_craw(url):
    # 주소 받아와서 html로 파싱
    html = urlopen(url).read()
    soup = BeautifulSoup(html, "html.parser")
    names = []

    # 태그 찾음
    source = soup.select('.mw-category-group > ul > li > a')

    # 배열로 만들어줌
    for name in source:
        name = str(name.text)
        name = name.split(' ')
        name = name[0]
        names.append(name)
    
    return names

# 여자 이름 크롤링
female_url = 'https://ko.wikipedia.org/wiki/%EB%B6%84%EB%A5%98:%EB%8C%80%ED%95%9C%EB%AF%BC%EA%B5%AD%EC%9D%98_%EC%97%AC%EC%9E%90_%EB%B0%B0%EA%B5%AC_%EC%84%A0%EC%88%98'
female_names = get_craw(female_url)
female_names_cnt = len(female_names)

# 남자 이름 크롤링
male_url = 'https://ko.wikipedia.org/wiki/%EB%B6%84%EB%A5%98:%EB%8C%80%ED%95%9C%EB%AF%BC%EA%B5%AD%EC%9D%98_%EB%82%A8%EC%9E%90_%EB%B0%B0%EA%B5%AC_%EC%84%A0%EC%88%98'
male_names = get_craw(male_url)
male_names_cnt = len(male_names)

users = []
names = []
genders = []
ages = []
blood_types = []
heights = []
weights = []
past_diagnostic_records = []
pub_dates = []
scores = []
pass_or_fails = []

# 임의로 데이터 삽입
for i in range(373):
    # 점수 난수 발생 -> 합불 평가
    score = random.randrange(0, 101)
    if(score >= 70):
        temp_pass = True
    else:
        temp_pass = False
    # 성별 난수 발생 -> 이름 선택
    gender = random.randrange(1, 3)
    users.append(1)
    if(gender == 1):
        names.append(male_names[random.randrange(0, male_names_cnt)])
    else:
        names.append(female_names[random.randrange(0, female_names_cnt)])
    blood_types.append(random.randrange(1, 5))
    heights.append(random.uniform(150, 190))
    weights.append(random.uniform(45, 120))
    past_diagnostic_records.append(random.randrange(1, 7))
    pub_dates.append("2019-06-04T15:43:28")
    scores.append(score)
    pass_or_fails.append(temp_pass)


set_field = ("Subject ID", "MRI ID", "Group", "Visit", "Visit", "M/F", "Hand", "Age","EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF")

# dataset 긁어오기
df_data = pd.read_csv('dataset.csv', encoding='euc-kr')

# 불필요한 속성값 제거
df_data = df_data.drop(columns="Subject ID")
df_data = df_data.drop(columns="MRI ID")
df_data = df_data.drop(columns="Group")
df_data = df_data.drop(columns="Visit")

# 문자열로 된 속성값 int로 변환 
# {'남' : 1 , '여' : 2}
df_data['M/F'] = df_data['M/F'].map({'M':1, 'F':2})
# {'오른속' : 1, '왼손' : 2}
df_data['Hand'] = df_data['Hand'].map({'R':1, 'L':2})

#결측값 찾기
#SES는 서수임으로 중간값인 1.0으로 대체
df_data['SES'] = df_data['SES'].fillna(1)
#MMSE는 mean으로 대체
df_data['MMSE'] = df_data['MMSE'].fillna(df_data['MMSE'].mean())

# print(df_data.isnull().sum())

# 속성이름 바꾸기
df_data.rename(columns={'M/F':'gender','Age':'age','EDUC':'educ','Hand':'hand','MR Delay':'MR_delay'}, inplace=True)


# 새로운 데이터 추가
df_data['user'] = users
df_data['name'] = names
df_data['blood_type'] = blood_types
df_data['height'] = heights
df_data['weight'] = weights
df_data['past_diagnostic_record'] = past_diagnostic_records
df_data['pub_date'] = pub_dates
df_data['score'] = scores
df_data['pass_or_fail'] = pass_or_fails

fieldname = []
for field in df_data.columns:
    fieldname.append(field)

fieldname = tuple(fieldname)
print(fieldname)

# # 인코딩 옵션으로 한글 깨지는거 해결!, index, 속성 필드 없애줌
df_data.to_csv('./seeddatatojson.csv', encoding='euc-kr',index=False, header=False)
df_data.to_csv('./seeddata.csv', encoding='euc-kr',index=False)

csvfile = open('seeddatatojson.csv', 'r', encoding='euc-kr')
jsonfile = open('seeddata.json', 'w')

reader = csv.DictReader(csvfile, fieldname)
jsonfile.write('[')
for row in reader:
    jsonfile.write('{"model": "scoring.scorelist","fields": ')
    json.dump(row, jsonfile, ensure_ascii = False)
    jsonfile.write('},\n')
jsonfile.write('{"model": "scoring.scorelist","fields": {"MR_delay": "1608", "gender": "2", "hand": "1", "age": "65", "educ": "13", "SES": "2.0", "MMSE": "30.0", "CDR": "0.0", "eTIV": "1333", "nWBV": "0.8009999999999999", "ASF": "1.317", "user": "1", "name": "이소영", "blood_type": "2", "height": "162.128979254895", "weight": "104.34307600829675", "past_diagnostic_record": "3", "pub_date": "2019-06-04T15:43:28", "score": "25", "pass_or_fail": "False"}}]')
