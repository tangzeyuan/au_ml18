import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm
def get_neighbors(data, p, eps=1.5):
    neighbors = []
    for x in data:
        if norm(x - p) < eps:
            if not np.array_equal(x, p):
                neighbors.append(x)
    return neighbors
 
def assign_labels(data, labels, p, k, eps, minPts):
    p_neighbors = get_neighbors(data, p, eps)
    i = 0
    while i < len(p_neighbors):
        pn = p_neighbors[i]
        index_pn = np.where(np.all(data==pn, axis=1))[0][0]
        if labels[index_pn] == 0:
            labels[index_pn] = k
        if labels[index_pn] == None:
            labels[index_pn] = k
            new_neighbors = get_neighbors(data, pn, eps)
            if len(new_neighbors) >= minPts:
                p_neighbors = np.vstack((p_neighbors, new_neighbors))
        i += 1

def my_DBSCAN(data, eps=1.5, minPts=3):
    labels = [None] * len(data)
    k = 1 # index for cluster label
    for index, o in enumerate(data):
        # o: object
        if labels[index] == None: # not classified object
            o_neighbors = get_neighbors(data, o, eps)
            if len(o_neighbors) >= minPts: # o is a core object
                assign_labels(data, labels, o, k, eps, minPts)
                k += 1
            else:
                labels[index] = 0 # o is not a core object, noise
    return labels






data = np.array([[1,3],
[2,3],
[4,1],
[4,4],
[5,2],
[5,5],
[5,6],
[6,1],
[5,1],
[6,3],
[6,2],
[5,3],
[4,2],
[4,5]])

labels = my_DBSCAN(data, 1.2, 2)
n_labels = len(np.unique(labels))
print(labels)
colors = ['C{}'.format(i) for i in range(n_labels)]

for l, c in zip(np.arange(n_labels), colors):
    x = data[np.where(labels == l)][:, 0]
    y = data[np.where(labels == l)][:, 1]
    print(x)
    plt.plot(x, y, 'o', color=c)
plt.show()

