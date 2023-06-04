from django.shortcuts import render,HttpResponse,redirect
import pandas as pd
import numpy as np
import re
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from dia_2.models import Peopleinfo
import pycaret
from pycaret.datasets import get_data
from pycaret.classification import *

def view(request):
    return render(request, 'index.html')
def get_user_input(request):
# Load and preprocess the diabetes dataset
    data = pd.read_csv('DIABpart2unc.csv')
# preprocess the dataset as needed
# Handle duplicates
    duplicate_rows_data = data[data.duplicated()]
    print("number of duplicate rows: ", duplicate_rows_data.shape)
    data = data.drop_duplicates()

    # Remove Unneccessary value [0.00195%]
    data = data[data['gender'] != 'Other']

    # Initialize the LabelBinarizer
    label_binarizer = LabelBinarizer()
    # Fit and transform the text column
    data['smoking_history'] = label_binarizer.fit_transform(data['smoking_history'])
     
    print(data.shape)
    print(data.columns)
    print(data.dtypes)
    print(data.nunique())
    print(data.isnull().sum())

    max_threshold=data.bmi.quantile(0.80)
    print(max_threshold)
    data=data[data.bmi<max_threshold]

    print(data.shape)
    print(data.diabetes.value_counts())
    
    x = np.array(data.drop(['diabetes'], axis=1))
    y=np.array(data.diabetes)
    #if axis =1 , drop a column
    #if axis =0 , drop a row 

# Split the data into training and testing sets
    x_train , x_test, y_train , y_test = train_test_split(x,y,test_size=0.25,random_state =12)

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)

    x_test = scaler.transform(x_test)

# PYCARET library
    exp = setup(data=data, target='diabetes',train_size=0.75, session_id=123)

    best_model = compare_models(fold=4)

    evaluate_model(best_model)

    tuned_model = tune_model(best_model)

    #Fiiting the tuned model 

    tuned_model.fit(x_train,y_train)

    tuned_model.score(x_test,y_test)
# Define a function to take user input

    if request.method == "POST":
        gender = float(request.POST.get('gender', 0))
        age = float(request.POST.get('age', 0))
        hypertension = float(request.POST.get('hypertension', 0))
        heart_disease = float(request.POST.get('heart_disease', 0))
        smoking_history = request.POST.get('smoking_history')
        height = float(request.POST.get('height', 0))
        weight = float(request.POST.get('weight', 0))
        HbA1c_level = float(request.POST.get('HbA1c_level', 0))
        if HbA1c_level == 0:
            HbA1c_level = float(data['HbA1c_level'].mean())
        blood_glucose_level = float(request.POST.get('blood_glucose_level', 0))
        if blood_glucose_level == 0:
            blood_glucose_level = float(data['blood_glucose_level'].mean())
    height=float(height)
    weight=float(weight)
    # Return None if the request method is not POST
    bmi = weight / (height * height)
# Call your input function to get user input
    user_input =  np.array([[gender, age, hypertension, heart_disease, smoking_history,bmi, HbA1c_level, blood_glucose_level]])
    prediction = tuned_model.predict(user_input)
    my_user=Peopleinfo(gender=gender, age=age,hypertension=hypertension, heart_disease=heart_disease,smoking_history=smoking_history, height=height ,weight=weight, HbA1c_level=HbA1c_level , blood_glucose_level=blood_glucose_level )
    my_user.save()
    if prediction == [0]:
        # return HttpResponse("The predicted outcome : you don't have diabetes")
        return render(request, 'diapredPositive.html')
    else:
        # return HttpResponse("The predicted outcome : you have diabetes") 
        return render(request, 'diapredNegative.html')
    
    
    

    



# Create your views here.
