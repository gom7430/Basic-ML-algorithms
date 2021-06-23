#---------------------------------------------------------------
#basic implementation of linear regression algorithm
#---------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#read data from a csv file
data = pd.read_csv("Advertising.csv")
print(data)

#visualize the data
plt.figure(figsize=(16,8))
scatterplot = plt.scatter(data ['TV'],data['sales'], c='black')
plt.show()

#convert the values into vectors
x=data['TV'].values.reshape(-1,1)
y=data['sales'].values.reshape(-1,1)

#implement linear regression model
regression_model = LinearRegression()
regression_model.fit(x,y)

#visualize and plot the best fit line
prediction_model = regression_model.predict(x)
plt.figure(figsize=(16,8))
scatterplot = plt.scatter(data ['TV'],data['sales'], c='black')

plt.plot(data['TV'],prediction_model,c= 'blue',linewidth = 2)
plt.xlabel('money spent on ads')
plt.ylabel('sales')
plt.show()