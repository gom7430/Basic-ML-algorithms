#---------------------------------------------------------------
#basic implementation of k-means clustering algorithm
#---------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#prepare the data
x = np.array([[5,3],[10,15],[15,12],[24,10],[30,45],[85,70],[71,80],[60,78],[55,52],[80,91]])

#visualize in a scatter plot
plt.scatter(x[:,0], x[:,1], label='true position')
plt.show()

#create clusters
kmeans = KMeans(n_clusters=2)
kmeans.fit(x)
print(kmeans.cluster_centers_)
print(kmeans.labels_)

#visualize clusters
plt.scatter(x[:,0],x[:,1],c=kmeans.labels_,cmap='Accent')
plt.show()

#try with 3 clusters
kmeans_three = KMeans(n_clusters = 3)
kmeans_three.fit(x)
print(kmeans_three.cluster_centers_)
print(kmeans_three.labels_)
plt.scatter(x[:,0],x[:,1],c=kmeans_three.labels_,cmap='rainbow')
plt.show()