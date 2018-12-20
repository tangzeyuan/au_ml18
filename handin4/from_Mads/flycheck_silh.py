from scipy.spatial.distance import cdist
from clustering_algorithms import compute_em_cluster, em_algorithm, lloyds_algorithm, tangs_algorithm

import numpy as np

def silhouette(data, clustering):
    n, d = data.shape
    k = np.unique(clustering)[-1]+1
 
    # YOUR CODE HERE
    #silh = None
    a = np.zeros(n)
    b = np.zeros(n)
    s = np.zeros(n)
    for i, o in enumerate(data):
        same_cluster = np.where(clustering == clustering[i])
        a[i] = np.sum(cdist(data[same_cluster], [o])) / len(same_cluster[0]) # same clusters
       
        tmp_b = []
        for j in range(k): 
            if j != clustering[i]: # loop over other clusters
                diff_cluster = np.where(clustering == j)
                tmp_b.append(np.sum(cdist(data[diff_cluster], [o])) / len(diff_cluster[0]))
        b[i] = min(tmp_b)
       
        s[i] = (b[i] - a[i]) / max(a[i], b[i])
    silh = np.sum(s) / n
    # END CODE
 
    return silh


if __name__ == '__main__':
    import sklearn.datasets
    iris = sklearn.datasets.load_iris()
    X = iris['data'][:,0:2] # reduce to 2d so you can plot if you want

    for k in range(2, 10):

        em_sc = 0
        lloyd_sc = 0 
        for jj in range(10):
            means, covs, probs_c, llh = em_algorithm(X, k, 50)
            clustering = compute_em_cluster(means, covs, probs_c, X)
            em_sc += silhouette(X, clustering)

        
            clustering, centroids, cost = lloyds_algorithm(X, k, 50)
            lloyd_sc += silhouette(X, clustering)

        print('Number of clusters {}: Lloyd: {}, EM {}'.format(k, lloyd_sc/10, em_sc/10))
