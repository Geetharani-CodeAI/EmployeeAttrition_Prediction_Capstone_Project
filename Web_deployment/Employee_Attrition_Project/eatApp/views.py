from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse, HttpRequest
from django.shortcuts import render, redirect
#from .forms import *
from django.contrib import messages
from django.shortcuts import render
from django.urls import reverse_lazy
from django.urls import reverse
from django.http import HttpResponse
from django.views.generic import (View,TemplateView,
ListView,DetailView,
CreateView,DeleteView,
UpdateView)
from . import models
from .forms import *
from django.core.files.storage import FileSystemStorage
#from topicApp.Topicfun import Topic
#from sklearn.tree import export_graphviz #plot tree
#from sklearn.metrics import roc_curve, auc #for model evaluation
#from sklearn.metrics import classification_report #for model evaluation
##from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(df2.drop('classification_yes', 1), df2['classification_yes'], test_size = .2, random_state=10)

import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
#from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
#from sklearn.decomposition import PCA
#from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import pickle
import matplotlib.pyplot as plt
#import eli5 #for purmutation importance
#from eli5.sklearn import PermutationImportance
#import shap #for SHAP values
#from pdpbox import pdp, info_plots #for partial plots
np.random.seed(123) #ensure reproduc
class dataUploadView(View):
    form_class = eatForm
    success_url = reverse_lazy('success')
    template_name = 'create.html'
    failure_url= reverse_lazy('fail')
    filenot_url= reverse_lazy('filenot')
    def get(self, request, *args, **kwargs):
        form = self.form_class()
        return render(request, self.template_name, {'form': form})
    def post(self, request, *args, **kwargs):
        #print('inside post')
        form = self.form_class(request.POST, request.FILES)
        #print('inside form')
        if form.is_valid():
            form.save()
            data_Age = request.POST.get('Age')
            data_DailyRate = request.POST.get('DailyRate')
            data_DistanceFromHome = request.POST.get('DistanceFromHome')
            data_Education = request.POST.get('Education')
            data_EnvironmentSatisfaction = request.POST.get('EnvironmentSatisfaction')
            data_HourlyRate = request.POST.get('HourlyRate')
            data_JobInvolvement = request.POST.get('JobInvolvement')
            data_JobLevel = request.POST.get('JobLevel')
            data_JobSatisfaction = request.POST.get('JobSatisfaction')
            data_MonthlyIncome = request.POST.get('MonthlyIncome')
            data_MonthlyRate = request.POST.get('MonthlyRate')
            data_NumCompaniesWorked = request.POST.get('NumCompaniesWorked')
            data_PercentSalaryHike = request.POST.get('PercentSalaryHike')
            data_PerformanceRating = request.POST.get('PerformanceRating')
            data_RelationshipSatisfaction = request.POST.get('RelationshipSatisfaction')
            data_StockOptionLevel = request.POST.get('StockOptionLevel')
            data_TotalWorkingYears = request.POST.get('TotalWorkingYears')
            data_TrainingTimesLastYear = request.POST.get('TrainingTimesLastYear')
            data_WorkLifeBalance = request.POST.get('WorkLifeBalance')
            data_YearsAtCompany = request.POST.get('YearsAtCompany')
            data_YearsInCurrentRole = request.POST.get('YearsInCurrentRole')
            data_YearsSinceLastPromotion = request.POST.get('YearsSinceLastPromotion')
            data_YearsWithCurrManager = request.POST.get('YearsWithCurrManager')
            data_BusinessTravel_Travel_Frequently = request.POST.get('BusinessTravel_Travel_Frequently')
            data_BusinessTravel_Travel_Rarely = request.POST.get('BusinessTravel_Travel_Rarely')
            data_Department_Research_and_Development = request.POST.get('Department_Research_and_Development')
            data_Department_Sales = request.POST.get('Department_Sales')
            data_EducationField_Life_Sciences = request.POST.get('EducationField_Life_Sciences')
            data_EducationField_Marketing = request.POST.get('EducationField_Marketing')
            data_EducationField_Medical = request.POST.get('EducationField_Medical')
            data_EducationField_Other = request.POST.get('EducationField_Other')
            data_EducationField_Technical_Degree = request.POST.get('EducationField_Technical_Degree')
            data_Gender_Male = request.POST.get('Gender_Male')
            data_JobRole_Human_Resources = request.POST.get('JobRole_Human_Resources')
            data_JobRole_Laboratory_Technician = request.POST.get('JobRole_Laboratory_Technician')
            data_JobRole_Manager = request.POST.get('JobRole_Manager')
            data_JobRole_Manufacturing_Director = request.POST.get('JobRole_Manufacturing_Director')
            data_JobRole_Research_Director = request.POST.get('JobRole_Research_Director')
            data_JobRole_Research_Scientist = request.POST.get('JobRole_Research_Scientist')
            data_JobRole_Sales_Executive = request.POST.get('JobRole_Sales_Executive')
            data_JobRole_Sales_Representative = request.POST.get('JobRole_Sales_Representative')
            data_MaritalStatus_Married = request.POST.get('MaritalStatus_Married')
            data_MaritalStatus_Single = request.POST.get('MaritalStatus_Single')
            data_OverTime_Yes = request.POST.get('OverTime_Yes')

            #print (data)
            dataset=pd.read_csv("Employee_Attrition_Dataset.csv",index_col=None)

            dataset = pd.get_dummies(dataset, dtype = int, drop_first = True)


            indep_x = dataset.drop("Attrition_Yes", axis =1)
            dep_y = dataset['Attrition_Yes']

            x_train, x_test, y_train, y_test = train_test_split(indep_x, dep_y, test_size = 0.30, random_state = 0)
            sc = StandardScaler()
            x_train = sc.fit_transform(x_train)
            x_test = sc.transform(x_test)

            classifier = LogisticRegression()
            classifier.fit(x_train, y_train)

            dicc={'yes':1,'no':0}
            filename = 'Final_model_Attrition.sav'
            classifier = pickle.load(open(filename, 'rb'))

            values = [float(data_Age),
                float(data_DailyRate),
                float(data_DistanceFromHome),
                float(data_Education),
                float(data_EnvironmentSatisfaction),
                float(data_HourlyRate),
                float(data_JobInvolvement),
                float(data_JobLevel),
                float(data_JobSatisfaction),
                float(data_MonthlyIncome),
                float(data_MonthlyRate),
                float(data_NumCompaniesWorked),
                float(data_PercentSalaryHike),
                float(data_PerformanceRating),
                float(data_RelationshipSatisfaction),
                float(data_StockOptionLevel),
                float(data_TotalWorkingYears),
                float(data_TrainingTimesLastYear),
                float(data_WorkLifeBalance),
                float(data_YearsAtCompany),
                float(data_YearsInCurrentRole),
                float(data_YearsSinceLastPromotion),
                float(data_YearsWithCurrManager),
                float(data_BusinessTravel_Travel_Frequently),
                float(data_BusinessTravel_Travel_Rarely),
                float(data_Department_Research_and_Development),
                float(data_Department_Sales),
                float(data_EducationField_Life_Sciences),
                float(data_EducationField_Marketing),
                float(data_EducationField_Medical),
                float(data_EducationField_Other),
                float(data_EducationField_Technical_Degree),
                float(data_Gender_Male),
                float(data_JobRole_Human_Resources),
                float(data_JobRole_Laboratory_Technician),
                float(data_JobRole_Manager),
                float(data_JobRole_Manufacturing_Director),
                float(data_JobRole_Research_Director),
                float(data_JobRole_Research_Scientist),
                float(data_JobRole_Sales_Executive),
                float(data_JobRole_Sales_Representative),
                float(data_MaritalStatus_Married),
                float(data_MaritalStatus_Single),
                float(data_OverTime_Yes)
            ]

            data = np.array(values).reshape(1, -1)
            
            #sc = StandardScaler()
            #data = sc.fit_transform(data.reshape(-1,1))
            out=classifier.predict(data.reshape(1,-1))
# providing an index
            #ser = pd.DataFrame(data, index =['bgr','bu','sc','pcv','wbc'])

            #ss=ser.T.squeeze()
#data_for_prediction = X_test1.iloc[0,:].astype(float)

#data_for_prediction =obj.pca(np.array(data_for_prediction),y_test)
            #obj=eat()
            ##plt.savefig("static/force_plot.png",dpi=150, bbox_inches='tight')







            return render(request, "succ_msg.html",{'data_Age': data_Age, 'data_DailyRate': data_DailyRate, 'data_DistanceFromHome': data_DistanceFromHome, 'data_Education': data_Education, 'data_EnvironmentSatisfaction': data_EnvironmentSatisfaction, 'data_HourlyRate': data_HourlyRate, 'data_JobInvolvement': data_JobInvolvement, 'data_JobLevel': data_JobLevel, 'data_JobSatisfaction': data_JobSatisfaction, 'data_MonthlyIncome': data_MonthlyIncome, 'data_MonthlyRate': data_MonthlyRate, 'data_NumCompaniesWorked': data_NumCompaniesWorked, 'data_PercentSalaryHike': data_PercentSalaryHike, 'data_PerformanceRating': data_PerformanceRating, 'data_RelationshipSatisfaction': data_RelationshipSatisfaction, 'data_StockOptionLevel': data_StockOptionLevel, 'data_TotalWorkingYears': data_TotalWorkingYears, 'data_TrainingTimesLastYear': data_TrainingTimesLastYear, 'data_WorkLifeBalance': data_WorkLifeBalance, 'data_YearsAtCompany': data_YearsAtCompany, 'data_YearsInCurrentRole': data_YearsInCurrentRole, 'data_YearsSinceLastPromotion': data_YearsSinceLastPromotion, 'data_YearsWithCurrManager': data_YearsWithCurrManager, 'data_BusinessTravel_Travel_Frequently': data_BusinessTravel_Travel_Frequently, 'data_BusinessTravel_Travel_Rarely': data_BusinessTravel_Travel_Rarely, 'data_Department_Research_and_Development': data_Department_Research_and_Development, 'data_Department_Sales': data_Department_Sales, 'data_EducationField_Life_Sciences': data_EducationField_Life_Sciences, 'data_EducationField_Marketing': data_EducationField_Marketing, 'data_EducationField_Medical': data_EducationField_Medical, 'data_EducationField_Other': data_EducationField_Other, 'data_EducationField_Technical_Degree': data_EducationField_Technical_Degree, 'data_Gender_Male': data_Gender_Male, 'data_JobRole_Human_Resources': data_JobRole_Human_Resources, 'data_JobRole_Laboratory_Technician': data_JobRole_Laboratory_Technician, 'data_JobRole_Manager': data_JobRole_Manager, 'data_JobRole_Manufacturing_Director': data_JobRole_Manufacturing_Director, 'data_JobRole_Research_Director': data_JobRole_Research_Director, 'data_JobRole_Research_Scientist': data_JobRole_Research_Scientist, 'data_JobRole_Sales_Executive': data_JobRole_Sales_Executive, 'data_JobRole_Sales_Representative': data_JobRole_Sales_Representative, 'data_MaritalStatus_Married': data_MaritalStatus_Married, 'data_MaritalStatus_Single': data_MaritalStatus_Single, 'data_OverTime_Yes': data_OverTime_Yes, 'out': out})



        else:
            return redirect(self.failure_url)
