import pandas as pd
# 데이터베이스에 있는 환자들에 대한 정보를 csv로 저장
def MakeCSV(patients):
    names = []
    genders = []
    ages = []
    ASFs = []
    nWBVs = []
    eTIVs = []
    CDRs = []
    MMSEs = []
    SESs = []
    educs = []
    MR_delays = []
    hands = []

    for patient in patients:
        names.append(patient['name'])
        genders.append(patient['gender'])
        ages.append(patient['age'])
        ASFs.append(patient['ASF'])
        nWBVs.append(patient['nWBV'])
        eTIVs.append(patient['eTIV'])
        CDRs.append(patient['CDR'])
        MMSEs.append(patient['MMSE'])
        SESs.append(patient['SES'])
        educs.append(patient['educ'])
        MR_delays.append(patient['MR_delay'])
        hands.append(patient['hand'])

    
    data = {
        "name" : names,
        "gender" : genders,
        "age" : ages,
        "ASF" : ASFs,
        "nWBV" : nWBVs,
        "eTIV" : eTIVs,
        "CDR" : CDRs,
        "MMSE": MMSEs,
        "SES": SESs,
        "educ": educs,
        "MR_delay": MR_delays,
        "hand": hands,
    }

    # 데이터 프레임 만들고 -> csv파일로 저장
    df = pd.DataFrame(data)
    # 인코딩 옵션으로 한글 깨지는거 해결!, index, 속성 필드 없애줌
    df.to_csv('./scoring/train.csv', encoding='euc-kr')
