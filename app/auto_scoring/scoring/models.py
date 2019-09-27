from django.db import models
from django.contrib.auth.models import User
from django.contrib import auth
from django.conf import settings

# Create your models here.
class ScoreList(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)
    title = models.CharField(max_length = 200)
    gender = models.CharField(max_length = 200)
    age = models.IntegerField()
    pub_date = models.DateTimeField('date published')

    def __str__(self):
        return self.title

class Photo(models.Model):
    scorelist = models.ForeignKey(ScoreList, on_delete=models.CASCADE, null=True)
    image = models.ImageField(upload_to='uploads/', blank=True, null=True)

class editPhoto(models.Model):
    scorelist = models.ForeignKey(ScoreList, on_delete=models.CASCADE, null=True)
    image = models.ImageField(upload_to='uploads/', blank=True, null=True)
