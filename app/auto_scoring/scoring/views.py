from django.shortcuts import render, get_object_or_404, redirect
from .models import ScoreList, Photo
from .forms import ScoreListForm
from django.utils import timezone
from .imageResizing import image_resizing
import os
from django.conf import settings


# Create your views here.
def home(request):
    return render(request, 'home.html')

def list(request):
    lists = ScoreList.objects.all()
    return render(request, 'list.html', {'lists':lists})

def scoring(request):
    if request.method == 'POST':
        form = ScoreListForm(request.POST)
        if form.is_valid():
            #모델 객체를 db에 저장하지 않은 상태로 반환
            list = form.save(commit=False)
            list.user = request.user
            list.pub_date = timezone.now()
            list.save()
            for i in range(1, 5):
                img = Photo()
                img.scorelist = list
                name = 'file' + str(i)
                check = 'check' + str(i)
                img.image = request.FILES[name]
                img.check = request.POST[check]
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
        return render(request, 'scoring.html', {'form':form, 'dic':dic})

def result(request, list_id):
    list = get_object_or_404(ScoreList, pk=list_id)
    return render(request, 'result.html', {'list' : list})

def analysis(request):
    lists = ScoreList.objects.all()
    age = {'30':0, '40':0, '50':0, '60':0, '70':0, '80':0}
    gender = {'남':0, '여':0}

    for patient in lists:
        patient_age = patient.age
        patient_gender = patient.gender

        if patient_gender == '남':
            gender_number = gender['남']
            gender['남'] = gender_number + 1
            
        else:
            gender_number = gender['여']
            gender['여'] = gender_number + 1
            

        if patient_age < 40:
            total = age['30']
            age['30'] = total + 1
        elif (patient_age >= 40) and (patient_age < 50):
            total = age['40']
            age['40'] = total + 1
        elif (patient_age >= 50) and (patient_age < 60):
            total = age['50']
            age['50'] = total + 1
        elif (patient_age >= 60) and (patient_age < 70):
            total = age['60']
            age['60'] = total + 1
        elif (patient_age >= 70) and (patient_age < 80):
            total = age['70']
            age['70'] = total + 1
        else :
            total = age['80']
            age['80'] = total + 1
        
    print(age)
    print(gender)
    return render(request, 'analysis.html', {'age':age, 'gender':gender})
