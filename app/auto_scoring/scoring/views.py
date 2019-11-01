from django.shortcuts import render, get_object_or_404, redirect
from .models import ScoreList, Photo, MRIAvg
from .forms import ScoreListForm
from django.utils import timezone
from .imageResizing import image_resizing
from .analysis import get_age, get_gender, get_educ
import os
from django.conf import settings
from django.core.paginator import Paginator, PageNotAnInteger
from .create_csv import MakeCSV
from .data_learning import learning_about_data, check_CDR
from django.utils import timezone
from .cnn import image_learning

# 메인화면
def home(request):
    return render(request, 'home.html')



# 검사자 리스트 화면
def list(request):
    subjects = ScoreList.objects.all().order_by('-id').filter(test=None)
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



def simple_test(request):
    if request.method == 'POST':
        list = ScoreList()
        list.name = 'test'
        list.test = True
        list.save()
        for i in range(1, 5):
            img = Photo()
            img.scorelist = list

            file_name = 'file' + str(i)
            check_name = 'check' + str(i)
            img.image = request.FILES[file_name]
            img.check = request.POST[check_name]
            img.grade = True
            print(img.image)
            img.save()
        return redirect('/scoring/simple_result/' + str(list.id))
    else:
        dic = {}
        for i in range(1, 5):
            file_name = 'file'+str(i)
            check_name = 'check'+str(i)
            dic[i] = check_name

    return render(request, 'simple_test.html', {'dic':dic})



def simple_result(request, list_id):
    list = get_object_or_404(ScoreList, pk=list_id)
    return render(request, 'simple_result.html', {'list' : list})




# 채점화면
def scoring(request):
    avg = MRIAvg.objects.last()
    if request.method == 'POST':
        check_CDR_object = {}
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
            
            if(request.POST['MRIcheck'] == 'no'):
                list.ASF = avg.avgASF
                list.nWBV = avg.avgnWBV
                list.eTIV = avg.avgeTIV
                list.MMSE = avg.avgMMSE
                list.SES = avg.avgSES
                list.MR_delay = avg.avgMR_delay
            else:
                list.ASF = request.POST['ASF']
                list.nWBV = request.POST['nWBV']
                list.eTIV = request.POST['eTIV']
                list.MMSE = request.POST['MMSE']
                list.SES = float(request.POST['SES'])
                list.MR_delay = int(request.POST["MR_Delay"])
            
            list.hand = int(request.POST['hand'])
            list.educ = request.POST['educ']

            # 임상 치매 여부 판단을 위한 딕셔너리
            check_CDR_object['gender'] = list.gender
            check_CDR_object['age'] = list.age
            check_CDR_object['ASF'] = list.ASF
            check_CDR_object['nWBV'] = list.nWBV
            check_CDR_object['eTIV'] = list.eTIV
            check_CDR_object['MMSE'] = list.MMSE
            check_CDR_object['SES'] = list.SES
            check_CDR_object['educ'] = list.educ
            check_CDR_object['hand'] = list.hand
            check_CDR_object['MR_delay'] = list.MR_delay

            list.CDR = check_CDR(check_CDR_object)
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
        return render(request, 'scoring.html', {'dic':dic, 'avg':avg})



# 결과화면
def result(request, list_id):
    list = get_object_or_404(ScoreList, pk=list_id)
    return render(request, 'result.html', {'list' : list})



# 데이터 분석 결과 화면
def analysis(request):
    subjects = ScoreList.objects.all().order_by('-id').filter(test=None)
    total_age = {50:0, 60:0, 70:0, 80:0, 90 : 0}
    total_age_cnt = {50:0, 60:0, 70:0, 80:0, 90 : 0}
    total_age_avg = {50:0, 60:0, 70:0, 80:0, 90 : 0}
    total = 0
    total_CDR = 0

    total_gender = {'male':0, 'female':0}
    total_gender_cnt = {'male':0, 'female':0}
    total_gender_avg = {'male':0, 'female':0}

    total_SES = {1:0, 2:0, 3:0, 4:0, 5:0}
    total_SES_avg = {1:0, 2:0, 3:0, 4:0, 5:0}

    total_educ = {8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0,15:0, 16:0, 17:0, 18:0, 19:0, 20:0}
    total_educ_cnt = {8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0,15:0, 16:0, 17:0, 18:0, 19:0, 20:0}
    total_educ_avg = {8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0,15:0, 16:0, 17:0, 18:0, 19:0, 20:0}

    for subject in subjects:
        total += 1
        total_CDR += subject.CDR

        total_educ[get_educ(subject.educ)] += subject.CDR
        total_educ_cnt[get_educ(subject.educ)] += 1

        total_age[get_age(subject.age)] += subject.CDR
        total_age_cnt[get_age(subject.age)] += 1

        total_gender[get_gender(subject.gender)] += subject.CDR
        total_gender_cnt[get_gender(subject.gender)] += 1

        total_SES[subject.SES] += subject.CDR
    
    for key in total_SES:
        total_SES_avg[key] = total_SES[key] / total_CDR
    
    for key in total_age:
        if total_age_cnt[key] == 0:
            continue
        total_age_avg[key] = total_age[key] / total_age_cnt[key]

    for key in total_gender:
        if total_gender_cnt[key] == 0:
            continue
        total_gender_avg[key] = total_gender[key] / total_gender_cnt[key]
    
    for key in total_educ:
        if total_educ_cnt[key] == 0:
            continue
        total_educ_avg[key] = total_educ[key] / total_educ_cnt[key]
    
    print(total_gender_avg)
    print(total_SES_avg)
    print(total_educ_avg)

    return render(request, 'analysis.html',{'total_educ_avg':total_educ_avg, 'total_SES_avg':total_SES_avg, 'total_age_avg':total_age_avg, 'total_gender_avg':total_gender_avg})



# 이미지 학습
def image_analysis(request):
    image_learning()
    return redirect('home')



# 데이터 학습
def data_analysis(request):
    # db에 저장된 환자 정보를 모드 긁어옴
    subjects = ScoreList.objects.all().order_by('-id').filter(test=None)
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
