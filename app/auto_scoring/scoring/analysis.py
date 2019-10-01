def get_age(age=0):
    if age < 40:
        return 30
    elif age >= 40 and age < 50:
        return 40
    elif age >= 50 and age < 60:
        return 50
    elif age >= 60 and age < 70:
        return 60
    elif age >= 70 and age < 80:
        return 70
    else:
        return 80

def get_gender(x=1):
    if x == 1:
        return 'male'
    else:
        return 'female'

def get_disease(x=1):
    if x == 1:
        return 'stroke'
    elif x == 2:
        return 'high_blood_pressure'
    elif x == 3:
        return 'heart_disease'
    elif x == 4:
        return 'diabetes'
    elif x == 5:
        return 'cancer'
    else:
        return 'none'
