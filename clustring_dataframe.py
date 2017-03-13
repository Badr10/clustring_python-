import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *


### For the purposes of this example, we store feature data from our
### dataframe `df`, in the `f1` and `f2` arrays. We combine this into
### a feature matrix `X` before entering it into the algorithm.
df = pd.read_csv('/Users/badrkhamis/Desktop/python_code/blocks_analysis/cls.csv')
# print df.head()

# f1 = df['Distance_Feature'].values
# f2 = df['Speeding_Feature'].values
#
# X=np.array(zip(df['fontposition'],df['topposition'],df['centerposition'],df['bolding']))
X=np.array(zip(df['LineStartX'],df['LineStartY'],df['fontposition'],df['topposition'],df['centerposition'],df['bolding']))
print len(X)
data_cluster = KMeans(n_clusters=6).fit(X)
label= data_cluster.labels_
centroids = data_cluster.cluster_centers_
colurs=['b.','r.','c.','k.','m.','y.','g.']
color = []; cluster = [];
for i in range(0,len(X)):

    color.append(colurs[label[i]])
    cluster.append(label[i])
    # print ('coordinate:',X[i], 'label:',label[i])
    plt.plot(X[i][0], X[i][1], colurs[label[i]], markersize=10)


cluster_data= pd.DataFrame(list(map(list, zip(cluster,color))), columns=['cluster','color'])
data= pd.concat([df,cluster_data],axis=1)
print data['Title_Ratio'].head(100)
plt.scatter(centroids[:,0],centroids[:,1], marker='x',s=150,linewidths=10,zorder=10)
# plt.show()
print data.cluster.value_counts()


