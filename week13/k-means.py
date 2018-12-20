import sklearn.datasets
iris = sklearn.datasets.load_iris()
X = iris['data'][:,0:2] # reduce dimensions so we can plot what happens.
k = 3
print(X.shape)

import numpy as np

def lloyds_algorithm(X, k, T):
    """ Clusters the data of X into k clusters using T iterations of Lloyd's algorithm. 
    
        Parameters
        ----------
        X : Data matrix of shape (n, d)
        k : Number of clusters.
        T : Maximum number of iterations to run Lloyd's algorithm. 
        
        Returns
        -------
        clustering: A vector of shape (n, ) where the i'th entry holds the cluster of X[i].
        centroids:  The centroids/average points of each cluster. 
        cost:       The cost of the clustering 
    """
    n, d = X.shape
    
    # Initialize clusters random. 
    clustering = np.random.randint(0, k, (n, )) 
    centroids  = np.zeros((k, d))
    
    # Used to stop if cost isn't improving (decreasing)
    cost = 0
    oldcost = 0
    
    # Column names
    print("Iterations\tCost")
    
    for i in range(T):
        
        # Update centroid
        
        # YOUR CODE HERE
        for c in range(k): # loop over clusters
            indices = np.where(clustering == c)
            #print(X[indices])
            print(np.sum(X[indices], axis=0))
            print(np.sum(X[indices], axis=0) / len(indices))
            centroids[c] = np.sum(X[indices], axis=0) / len(X[indices])
        # END CODE

        
        # Update clustering 
        
        # YOUR CODE HERE
        for x_i, x in enumerate(X): # loop over data
            j = 0
            d = np.inf
            for c in range(k): # loop over clusters
                if np.linalg.norm(x - centroids[c]) < d:
                    j = c
                    d = np.linalg.norm(x - centroids[c]) ** 2
            clustering[x_i] = j
        # END CODE
        
        
        # Compute and print cost
        cost = 0
        for j in range(n):
            cost += np.linalg.norm(X[j] - centroids[clustering[j]])**2    
        print(i+1, "\t\t", cost)
        # fast alternative: cost = np.sum((X - centroids[clustering])**2) 
        
        # Stop if cost didn't improve more than epislon (decrease)
        if np.isclose(cost, oldcost): break #TODO
        oldcost = cost
        
    return clustering, centroids, cost

clustering, centroids, cost = lloyds_algorithm(X, 3, 100)
#print(X)
print(clustering)