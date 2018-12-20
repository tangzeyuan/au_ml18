import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm
def get_neighbors(data, p, eps=1.5):
    neighbors = []
    for x in data:
        if norm(x - p) < eps:
            if not np.array_equal(x, p):
                neighbors.append(x)
    return np.array(neighbors)

def my_DBSCAN(data, eps=1.5, minPts=3):
    c = 0
    labels = {tuple(o): None for o in data}
    for o in data:
        o = tuple(o)
        if labels[o] != None: 
            continue
        
        neighbors = get_neighbors(data, o, eps)
        if len(neighbors) < minPts:
            labels[o] = c # noise
            continue
        
        c += 1
        labels[o] = c
        for x in neighbors:
            x = tuple(x)
            if labels[x] == 0:
                labels[x] = c
            elif labels[x] != None:
                continue
            else:
                labels[x] = c
                new_neighbors = get_neighbors(data, x, eps)
                if len(new_neighbors) >= minPts:
                    i = 0
                    while i < len(neighbors):
                        n = neighbors[i]
                        if not np.any(neighbors[:,0] == n):
                            print(n)
                            neighbors = np.append(neighbors, n, axis=0)

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

print(get_neighbors(data, data[0]))
for key, value in my_DBSCAN(data).items():
    print(key, value)
plt.plot(data[:,0], data[:,1], 'bo')
plt.show()

