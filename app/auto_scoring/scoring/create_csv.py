import pandas as pd

def MakeCSV(patients):
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

    for patient in patients:
        users.append(patient['user'])
        names.append(patient['name'])
        genders.append(patient['gender'])
        ages.append(patient['age'])
        blood_types.append(patient['blood_type'])
        heights.append(patient['height'])
        weights.append(patient['weight'])
        past_diagnostic_records.append(patient['past_diagnostic_record'])
        pub_dates.append(patient['pub_date'])
        scores.append(patient['score'])
        pass_or_fails.append(patient['pass_or_fail'])

    
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
    df.to_csv('./scoring/train.csv', encoding='euc-kr')
