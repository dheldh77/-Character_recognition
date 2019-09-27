from django import forms
from .models import ScoreList

# 만약 모델 기반이 아니라면 forms.Form
class ScoreListForm(forms.ModelForm):
    class Meta:
        model = ScoreList
        fields = ['title','gender','age']
