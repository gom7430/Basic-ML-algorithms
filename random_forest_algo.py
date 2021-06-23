#---------------------------------------------------------------
#basic implementation of random forest algorithm
#---------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#read dataset
iris_data = pd.read_csv('Iris.csv')
x = iris_data.drop('Class',axis = 1)
y = iris_data['Class']

#split training and test data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20,random_state = 0)
print(x_train.shape)
print(y_train.shape)

#apply normalizing of features in the data
scaler_data = StandardScaler()
x_train = scaler_data.fit_transform(x_train)
x_test = scaler_data.fit_transform(x_test)

#implement the algorithm
classifier = RandomForestClassifier()
classifier.fit(x_train,y_train)

#predict the values for new data
y_pred = classifier.predict(x_test)
print(y_pred)

#evaluate the algorithm
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))