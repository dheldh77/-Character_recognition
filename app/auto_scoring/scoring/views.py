from django.shortcuts import render, get_object_or_404, redirect
from .models import ScoreList, Photo, MRIAvg
from .forms import ScoreListForm
from django.utils import timezone
from .imageResizing import image_resizing
from .analysis import get_age, get_gender, get_disease
import os
from django.conf import settings
from django.core.paginator import Paginator, PageNotAnInteger
from .create_csv import MakeCSV
from .data_learning import learning_about_data
from django.utils import timezone

# 메인화면
def home(request):
    return render(request, 'home.html')



# 검사자 리스트 화면
def list(request):
    subjects = ScoreList.objects.all().order_by('-id')
    # 페이지네이션
    paginator = Paginator(subjects, 10)
    try:
        page = request.GET.get('page', 1)
    except PageNotAnInteger:
        page = 1
    
    lists = paginator.get_page(page)
    return render(request, 'list.html', {'lists':lists})



# 선택화면
def select(request):
    return render(request, 'select.html')



# 채점화면
def scoring(request):
    if request.method == 'POST':
        form = ScoreListForm(request.POST)
        if form.is_valid():
            #모델 객체를 db에 저장하지 않은 상태로 반환
            list = form.save(commit=False)
            list.user = request.user
            list.name = request.POST['name']
            list.gender = int(request.POST['gender'])
            list.age = request.POST['age']
            list.blood_type = int(request.POST['blood_type'])
            list.height = request.POST['height']
            list.weight = request.POST['weight']
            list.past_diagnostic_record = int(request.POST['past_diagnostic_record'])
            list.pub_date = request.POST['pub_date']
            list.score = 30
            if list.score >= 70:
                list.pass_or_fail = True
            else:
                list.pass_or_fail = False
            list.ASF = request.POST['ASF']
            list.nWBV = request.POST['nWBV']
            list.eTIV = request.POST['eTIV']
            list.hand = int(request.POST['hand'])
            list.MMSE = request.POST['MMSE']
            list.SES = float(request.POST['SES'])
            list.educ = request.POST['educ']
            list.MR_delay = int(request.POST["MR_Delay"])
            list.CDR = 1.0
            list.save()

            # 이미지 저장
            for i in range(1, 5):
                img = Photo()
                img.scorelist = list
                file_name = 'file' + str(i)
                check_name = 'check' + str(i)
                img.image = request.FILES[file_name]
                img.check = request.POST[check_name]
                img.grade = True
                img.save()
            # for afile in request.FILES.getlist('file'):
            #     img = Photo()
            #     img.scorelist = list
            #     img.image = afile
            #     img.save()
            #     image_resizing(img.image.path,img.image.name)

            return redirect('/scoring/result/' + str(list.id))
    else:
        form = ScoreListForm()
        dic = {}
        for i in range(1, 5):
            file_name = 'file'+str(i)
            check_name = 'check'+str(i)
            dic[i] = check_name
        return render(request, 'scoring.html', {'dic':dic})



# 결과화면
def result(request, list_id):
    list = get_object_or_404(ScoreList, pk=list_id)
    return render(request, 'result.html', {'list' : list})



# 데이터 분석 결과 화면
def analysis(request):
    lists = ScoreList.objects.all()
    total = 0
    patient = 0
    # age [30대 이하, 40대, 50대, 60대, 70대, 80대 이상]
    age_total = { 30:0, 40:0, 50:0, 60:0, 70:0, 80:0} # 전체
    age_patient = { 30:0, 40:0, 50:0, 60:0, 70:0, 80:0} # 환자
    age_rate = { 30:0.0, 40:0.0, 50:0.0, 60:0.0, 70:0, 80:0.0} # 비율
    # gender ['남', '여']
    gender_total = {'male':0, 'female':0} # 전체
    gender_patient = {'male':0, 'female':0} # 환자
    gender_rate = {'male':0.0, 'female':0.0} # 비율
    # past_diagnostic_record [stroke, high_blood_pressure, heart_disease, diabetes, cancer, none]
    disease_patient = {'stroke':0, 'high_blood_pressure':0, 'heart_disease':0, 'diabetes':0, 'cancer':0, 'none':0}

    for subject in lists:
        # 전수 조사
        total += 1
        age_total[get_age(subject.age)] += 1
        gender_total[get_gender(subject.gender)] +=1

        # 치매 진단 환자일 경우
        if(subject.pass_or_fail == False):
            patient += 1
            age_patient[get_age(subject.age)] += 1
            gender_patient[get_gender(subject.gender)] +=1
            disease_patient[get_disease(subject.past_diagnostic_record)] += 1
    
    for key in age_total:
        if age_total[key] == 0.0:
            age_rate[key] = 0.0
            continue
        age_rate[key] = age_patient[key] / age_total[key]

    for key in gender_total:
        if gender_total[key] == 0.0:
            gender_rate[key] = 0.0
            continue
        gender_rate[key] = gender_patient[key] / gender_total[key]
    
    print_patient = [patient]
    print(print_patient)

    return render(request, 'analysis.html',{'gender_rate':gender_rate, 'age_rate':age_rate, 'disease_patient':disease_patient, 'patients':print_patient})



# 이미지 학습
def image_analysis(request):
    return redirect('home')



# 데이터 학습
def data_analysis(request):
    # db에 저장된 환자 정보를 모드 긁어옴
    subjects = ScoreList.objects.all().order_by('-id')
    list_subjects = []

    for subject in subjects:
        subject_dic = {}
        # subject_dic["user"] = subject.user.username
        subject_dic["name"] = subject.name
        subject_dic["gender"] = subject.gender
        subject_dic["age"] = subject.age
        # subject_dic["blood_type"] = subject.blood_type
        # subject_dic["height"] = subject.height
        # subject_dic["weight"] = subject.weight
        # subject_dic["past_diagnostic_record"] = subject.past_diagnostic_record
        # subject_dic["pub_date"] = subject.pub_date
        # subject_dic["score"] = subject.score
        # subject_dic["pass_or_fail"] = subject.pass_or_fail

        subject_dic["ASF"] = subject.ASF
        subject_dic["nWBV"] = subject.nWBV
        subject_dic["eTIV"] = subject.eTIV
        subject_dic["CDR"] = subject.CDR
        subject_dic["MMSE"] = subject.MMSE
        subject_dic["SES"] = subject.SES
        subject_dic["educ"] = subject.educ
        subject_dic["MR_delay"] = subject.MR_delay
        subject_dic["hand"] = subject.hand
        list_subjects.append(subject_dic)
    MakeCSV(list_subjects)
    
    mriavg = MRIAvg()
    avg = {}
    # 학습된 모델 생성
    avg = learning_about_data()
    mriavg.avgASF = avg['ASF']
    mriavg.avgeTIV = avg['eTIV']
    mriavg.avgMMSE = avg['MMSE']
    mriavg.avgMR_delay = avg['MR_delay']
    mriavg.avgnWBV = avg['nWBV']
    mriavg.avgSES = avg['SES']
    mriavg.analysis_date = timezone.now()
    mriavg.save()
    return redirect('analysis')
