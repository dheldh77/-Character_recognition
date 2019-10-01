from django.db import models
from django.contrib.auth.models import User
from django.contrib import auth
from django.conf import settings

# Create your models here.
class ScoreList(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True) # 담당의
    title = models.CharField(max_length = 200) # 환자이름
    gender = models.CharField(max_length = 200) # 성별
    age = models.IntegerField() # 나이
    pub_date = models.DateTimeField('date published') # 제출일
    score = models.IntegerField(blank=True) # 점수
    pass_or_fail = models.BooleanField(blank=True) # 침애여부




    def __str__(self):
        return self.title

class Photo(models.Model):
    scorelist = models.ForeignKey(ScoreList, on_delete=models.CASCADE, null=True) # 외래키
    image = models.ImageField(upload_to='uploads/', blank=True, null=True) # 이미지
    check = models.IntegerField(blank=True) # 정답
    grade = models.BooleanField(blank=True) # 합격 여부
