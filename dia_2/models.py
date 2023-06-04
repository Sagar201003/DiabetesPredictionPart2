from django.core.exceptions import ValidationError
from django.shortcuts import redirect
from django.http import request
from django.db import models
from django.contrib import messages
# # Create your models here.
class Peopleinfo(models.Model):
    gender=models.FloatField(unique=False)
    age=models.FloatField(unique=False)
    #name=models.CharField(max_length=100,null=False,blank=False,unique=False)
    hypertension =models.FloatField(unique=False)
    heart_disease=models.FloatField(unique=False) 
    # Phone=models.CharField(max_length=10,unique=False)
    smoking_history=models.FloatField(unique=False) 
    height=models.FloatField(unique=False)
    weight=models.FloatField(unique=False)
    HbA1c_level=models.FloatField(unique=False)
    blood_glucose_level=models.FloatField(unique=False)
    

# Create your models here.
