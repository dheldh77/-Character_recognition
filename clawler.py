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

# 여자 250명

# 남자 500명
for i in range(500):
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
    genders.append(gender)
    ages.append(random.randrange(15, 100))
    blood_types.append(random.randrange(1, 5))
    heights.append(random.uniform(150, 190))
    weights.append(random.uniform(45, 120))
    past_diagnostic_records.append(random.randrange(1, 7))
    pub_dates.append("2019-06-04T15:43:28")
    scores.append(score)
    pass_or_fails.append(temp_pass)



data = {
    "user" : users,
    "name" : names,
    "gender" : genders,
    "age" : ages,
    "blood_type" : blood_types,
    "height" : heights,
    "weight" : weights,
    "past_diagnostic_record" : past_diagnostic_records,
    "pub_date": pub_dates,
    "score": scores,
    "pass_or_fail": pass_or_fails
}

# 데이터 프레임 만들고 -> csv파일로 저장
df = pd.DataFrame(data)
# 인코딩 옵션으로 한글 깨지는거 해결!, index, 속성 필드 없애줌
df.to_csv('./seeddata.csv', encoding='euc-kr',index=False, header=False)

csvfile = open('seeddata.csv', 'r', encoding='euc-kr')
jsonfile = open('seeddata.json', 'w')

fieldnames = ("user", "name", "gender", "age", "blood_type", "height", "weight", "past_diagnostic_record", "pub_date", "score", "pass_or_fail")
reader = csv.DictReader( csvfile, fieldnames)
for row in reader:
    json.dump(row, jsonfile, ensure_ascii = False)
    jsonfile.write(',\n')
