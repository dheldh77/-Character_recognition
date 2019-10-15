from django.contrib import admin
from .models import ScoreList, Photo, MRIAvg

class PicInline(admin.TabularInline):
    model = Photo

class ImageAdmin(admin.ModelAdmin):
    inlines = [PicInline,]
# Register your models here.
admin.site.register(ScoreList, ImageAdmin)
admin.site.register(MRIAvg)
