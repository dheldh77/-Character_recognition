def get_age(age=0):
    if age < 60:
        return 50
    elif age >= 60 and age < 70:
        return 60
    elif age >= 70 and age < 80:
        return 70
    elif age >= 80 and age < 90:
        return 80
    else:
        return 90

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

def get_educ(x=8):
    if x <= 8:
        return 8
    elif x >= 20:
        return 20
    else:
        return x
