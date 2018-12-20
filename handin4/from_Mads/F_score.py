import numpy as np

def f1(predicted, labels):
    n, = predicted.shape
    assert labels.shape == (n,)
    r = np.max(predicted) + 1
    k = np.max(labels) + 1

    # Implement the F1 score here
    # YOUR CODE HERE
    contingency = np.zeros((r, k))
    for i in range(r):
        for j in range(k):
            I = np.argwhere(predicted==i).flatten()
            J = np.argwhere(labels==j).flatten()
            contingency[i, j] = len(np.intersect1d(I, J))
            

    m = np.sum(contingency, axis=0)
    n = np.sum(contingency, axis=1)

    prec = np.max(contingency, axis=1)/n
    recall = np.max(contingency, axis=1)/m[np.argmax(contingency, axis=1)]

    


    
    #for i in range(r):
    #    j = np.argmax(contingency[i])
        
        

    
#    print(contingency.shape)
#    print(n.shape)
#    print(m.shape)
    

#    prec = np.max(contingency, axis=1)/n

    F_individual = 2 * prec * recall / ( prec + recall )
    F_overall = np.sum(F_individual)/r
    # END CODE

    assert contingency.shape == (r, k)
    return F_individual, F_overall, contingency

if __name__ == '__main__':
    from clustering_algorithms import em_algorithm, compute_em_cluster, lloyds_algorithm

    # Load the Iris data set
    import sklearn.datasets
    X, y = sklearn.datasets.load_iris(True)
    X = X[:,0:2] # reduce to 2d so you can plot if you want
    print(X.shape, y.shape)

    for k in range(2, 10):
        means, covs, probs_c, llh = em_algorithm(X, k, 100)
        em_cluster = compute_em_cluster(means, covs, probs_c, X)
        _, em_sc, _ = f1(em_cluster, y)
        p, _, _ = lloyds_algorithm(X, k, 50)
        _, lloyd_sc, _ = f1(p, y)
        print('Cluster size {}: EM F1 = {}, Lloyd F1 = {}'.format(k, em_sc, lloyd_sc))
    



    
    
