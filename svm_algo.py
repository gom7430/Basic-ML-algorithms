#---------------------------------------------------------------
#basic implementation of Support Vector Classifier
#-------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

#read dataset
iris_data = pd.read_csv("Iris.csv")
print(iris_data)

#analyse the dataset
print(iris_data.shape)

#data preprocessing to divide attributes and labels
x = iris_data.drop('Class',axis = 1)
y = iris_data['Class']

#divide training and testing dataset
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20)
print(x_train.shape)
print(y_train.shape)

#implement the svc algorithm
svclassifier = SVC(kernel='linear')
svclassifier.fit(x_train,y_train)

#predicting the output classes
y_predict = svclassifier.predict(x_test)
print(y_predict)

#evaluate the model
svc_evaluate = classification_report(y_test,y_predict)
print(svc_evaluate)