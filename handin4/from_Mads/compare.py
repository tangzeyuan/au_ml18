from clustering_algorithms import lloyds_algorithm, em_algorithm, compute_em_cluster
import matplotlib.pyplot as plt
import numpy as np

# Load the Iris data set
import sklearn.datasets
X, y = sklearn.datasets.load_iris(True)
X = X[:,0:2] # reduce to 2d so you can plot if you want


# Make figure:
fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
axes = axes.flatten()

titles = ["Lloyd's", "EM", 'Data']

k = 3
for i, ax in enumerate(axes):
# Cluster data:
    if i == 0:
        clustering, means, cost = lloyds_algorithm(X, k, 100)
        
    if i == 1:
        means, covs, probs_c, llh = em_algorithm(X, k, 100)
        clustering = compute_em_cluster(means, covs, probs_c, X)

    if i == 2:
        clustering = y
        means = np.zeros(k*2).reshape(k, 2)


    for jj in range(k):
        mask = clustering == jj
        l1 = ax.scatter(X[mask, 0], X[mask, 1])

        if i != 2:
            ax.scatter(means[jj, 0], means[jj, 1], c=l1.get_facecolor(), marker='x')

        ax.set_title(titles[i])


plt.tight_layout()
plt.savefig('algo_compare.png')
plt.show()







