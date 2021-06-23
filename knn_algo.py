#---------------------------------------------------------------
#basic implementation of K-NN algorithm
#---------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

#read data
iris_data = pd.read_csv('Iris.csv')
print(iris_data)

#split attributes and label in the data
x = iris_data.drop('Class',axis = 1)
y = iris_data['Class']

#split data into training and test data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20)
print(x_train.shape)
print(y_train.shape)

#preprocessing data and normalizing features
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#implement KNN algorithm
classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(x_train,y_train)

#predict for new data
y_pred = classifier.predict(x_test)
print(y_pred)

#evaluate the algorithm
print (classification_report(y_test,y_pred))
