# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.http import JsonResponse,HttpResponse
from sklearn.svm import SVC

from django.shortcuts import render
import numpy as np
from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split
import csv
import json
#import urllib.request

def Algo(request):
 df_x = pd.read_csv("mycsv.csv", header=0)
 df_y = pd.read_csv("Book2.csv", header=0)

 X = df_x.iloc[:,:].values

 Y = df_y.iloc[:,:].values

#X(predictor) and Y(target) for training set and x_test(predictor) of test_dataset
#create SVM classification object

 X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33)

 model=svm.SVC(kernel='linear',C=10,gamma=1)
 model = SVC(kernel = 'linear' , random_state=2)
 model.fit(X_train,Y_train)
 print ('score Scikit learn: ', model.score(X_test,Y_test))

#train the model
 model.fit(X,Y)
 model.score(X,Y)

 #decoding data
#  urls=""
#  response=urllib.request.urlopen(urls)
#  body_data=response.read()
#  if request.META.get('CONTENT_TYPE', '').lower() == 'application/json' and len(request.body) > 0:
#     try:
#     	body_data=json.loads(body_data.decode("utf8"))
#     except Exception as e:
#         return HttpResponseBadRequest(json.dumps({'error': 'Invalid request: {0}'.format(str(e))}), content_type="application/json")

# #predict Output
#  xtest=body_data
 x_test=[[0,0,0,1,0,0,0],[1,0,0,0,0,1,1]] #test ip
 predicted=model.predict(x_test)
 print predicted
 print model.score(X_test,Y_test)

 info={'answer':predicted[0],'accuracy':model.score(X_test,Y_test)}

 # print info
 return JsonResponse(info, safe=False)
 #return HttpResponse(result)


# Create your views here.

