from django.contrib import admin
from .models import Peopleinfo
class AdminPeopleinfo(admin.ModelAdmin):
    list_display= ['gender','age','hypertension','heart_disease','smoking_history','height','weight','HbA1c_level','blood_glucose_level']
# Register your models here.
admin.site.register(Peopleinfo,AdminPeopleinfo)