#!/usr/bin/env python
# coding: utf-8

# In[86]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

sigma = [0.5, 1, 2, 4, 8]
kmeans_scores = []
gmm_scores = []
gmm_acc = []
kmeans_acc = []
for s in sigma:
    sample1 = np.random.multivariate_normal([-1, -1], np.dot(s,[[2, 0.5], [0.5, 1]]), 100)
    sample2 = np.random.multivariate_normal([1, -1], np.dot(s,[[1, -0.5], [-0.5, 2]]), 100)
    sample3 = np.random.multivariate_normal( [0, 1] , np.dot(s,[[1, 0], [0, 2]]), 100)
    Total_sample = np.concatenate((sample1, sample2, sample3))

    m = 0
    kmeans = KMeans(n_clusters=3, init='k-means++', algorithm="lloyd").fit(Total_sample)
    kmeans_scores.append(kmeans.score(Total_sample))
    for idx, label in enumerate(kmeans.labels_):
        if idx < 100:
            if label != np.bincount(kmeans.labels_[:100]).argmax():
                m = m+1
        if  100<= idx<200:
            if label != np.bincount(kmeans.labels_[100:200]).argmax():
                m = m+1
        if 200 <= idx:
            if label != np.bincount(kmeans.labels_[200:]).argmax():
                m = m+1
    acc = 1 - (m/len(Total_sample))
    kmeans_acc.append(acc)
    
    gmm = GaussianMixture(n_components=3).fit(Total_sample)
    gmm_scores.append(GMM.score(Total_sample))
    gmm_labels = []
    
    for x in Total_sample:
        d = []
        for i in range(3):
            d.append(np.sqrt((x[0] - gmm.means_[i][0])**2 + (x[1] - gmm.means_[i][1])**2))
        gmm_labels.append(np.argmin(d))

    r =0
    for idx, label in enumerate(gmm_labels):
        if idx < 100:
            if label != np.bincount(gmm_labels[:100]).argmax():
                r = r+1
        if 100<= idx < 200:
            if label != np.bincount(gmm_labels[100:200]).argmax():
                r = r+1
        if 200 <= idx:
            if label != np.bincount(gmm_labels[200:]).argmax():
                r= r+1
    
    acc = 1 - (r/len(Total_sample))
    gmm_acc.append(acc) 

plt.plot(sigma, kmeans_acc)
plt.ylabel('Accuracy')
plt.xlabel('Sigma')
plt.title('Kmeans Clustering Accuracy')
plt.show()

plt.plot(sigma, gmm_acc)
plt.ylabel('Accuracy')
plt.xlabel('Sigma')
plt.title('GMM Clustering Accuracy')
plt.show()

plt.plot(sigma, kmeans_scores)
plt.title('Kmeans Scores')
plt.ylabel('Score')
plt.xlabel('Sigma')
plt.show()

plt.plot(sigma, gmm_scores)
plt.title('GMM scores')
plt.ylabel('Score')
plt.xlabel('Sigma')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




