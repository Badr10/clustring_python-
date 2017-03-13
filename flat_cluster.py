# from
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
X= np.array([[1,2,3],
            [5,8,8],
            [1.5,1.8,1.6],
            [8,8,7],
            [1,0.6,0.9],
            [9,11,10],
             [12,14,13]])
dd = KMeans(n_clusters=3)
dd.fit(X)
centriods = dd.cluster_centers_
labels = dd.labels_

# print centriods
print labels
print dd.inertia_
# print dd.score(X, y=None)

colurs=['g.','r.','y.']
label = [] ; value = []
    # for i in range(len(X)):
for s , i in [('Setosa', 0),
                    ('Versicolour', 1),
                    ('Virginica', 2)]:
    value.append(X[i])
    print centriods
    # centrioid.append(centriods[i][0])
    label.append(labels[i])
    # print ('coordinate:',X[i], 'label:',labels[i])
    plt.plot(X[i][0], X[i][1], colurs[labels[i]], markersize = 20)
# cluster_data = pd.DataFrame(
#     {'cluster': label,
#      'value': value

     # })
cluster_data= pd.DataFrame(list(map(list, zip(label,value))), columns=['cluster','values'])
plt.scatter(centriods[:,0],centriods[:,1], marker='*',s=150,linewidths=10,zorder=10)
plt.show()
# print label
# print cluster_data.head()
print dd.cluster_centers_.squeeze().mean()

