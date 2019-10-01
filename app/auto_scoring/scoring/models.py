from django.db import models
from django.contrib.auth.models import User
from django.contrib import auth
from django.conf import settings

# Create your models here.
class ScoreList(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True) # 담당의
    name = models.CharField(max_length=200,null=True) # 성명
    gender = models.IntegerField(null=True) # 성별
    age = models.IntegerField(null=True) # 연령
    blood_type = models.IntegerField(null=True) # 혈액형
    height = models.FloatField(null=True) # 신장
    weight = models.FloatField(null=True) # 몸무게
    past_diagnostic_record = models.IntegerField(null=True) # 과거력
    pub_date = models.DateTimeField(null=True) # 진달일
    score = models.IntegerField(null=True) # 점수
    pass_or_fail = models.BooleanField(null=True) # 치매여부

    def __str__(self):
        return self.name

class Photo(models.Model):
    scorelist = models.ForeignKey(ScoreList, on_delete=models.CASCADE, null=True) # 외래키
    image = models.ImageField(upload_to='uploads/', blank=True, null=True) # 이미지
    check = models.IntegerField(blank=True, null=True) # 정답
    grade = models.BooleanField(blank=True, null=True) # 합격 여부
