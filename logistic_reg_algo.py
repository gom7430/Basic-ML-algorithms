#---------------------------------------------------------------
#basic implementation of logistic regression algorithm
#---------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

#define a dataset
x = np.array([100,120,150,180,200,200,202,203,205,210,215,250,270,300,305,310])
y = np.array([1,1,1,1,1,1,1,0,1,0,0,0,0,0,0,0])

#plot the data
plt.scatter(x,y)
plt.title('pricing bids')
plt.xlabel('bids')
plt.ylabel('gain or loss gain:1,loss:0')
plt.show()

#implement the model
logreg_model = LogisticRegression(C=1.0,solver='lbfgs',multi_class='ovr')
X= x.reshape(-1,1)
logreg_model.fit(X,y)

#predict with new data
print(logreg_model.predict([[111]]))
print(logreg_model.predict([[285]]))