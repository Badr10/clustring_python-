from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
df = pd.read_csv('/Users/badrkhamis/Desktop/python_code/blocks_analysis/cls.csv')


new_data=df[['LineStartX','LineStartY','fontposition','topposition','centerposition','bolding']]
print new_data.head()
number_of_cluster=range(1,10)
distance_between_clustre=[]
print new_data.shape[0]
for k in number_of_cluster:
    model=KMeans(n_clusters=k)
    model.fit(new_data)
    #using Elbow Method to choose the best number of cluster .
    distance_between_clustre.append(sum(np.min(cdist(new_data, model.cluster_centers_, 'euclidean'), axis=1))
    / new_data.shape[0])
print distance_between_clustre
plt.plot(number_of_cluster, distance_between_clustre)
plt.xlabel('Number of clusters')
plt.ylabel('Average distance')
plt.title('Selecting k with the Elbow Method')
# plt.show()