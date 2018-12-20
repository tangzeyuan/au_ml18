import numpy as np
from sklearn.preprocessing import StandardScaler

def test1():
    a = np.arange(20).reshape(5, 4)
    print(a)
    print(a[0, :])
    print(a[:, 0])
    print(np.mean(a))
    print(np.mean(a, axis=0))
    print(np.mean(a, axis=1))
    print('\n')

def test2():
    np.random.seed(1)
    X = np.random.randint(5, size=20).reshape(5, 4)

    s = StandardScaler()
    X_std = s.fit_transform(X)
    print(X)
    print(X_std)
    print(s.mean_)
    print(s.var_)
    print('\n')

    mean_vec = np.mean(X_std, axis=0)
    cov_mat = np.cov(X_std.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    print(mean_vec)
    print(cov_mat)
    print(eig_vals)
    print(eig_vecs)


def test3():
    X1 = np.arange(10)
    Y11 = 3 * X1 + 8
    Y12 = -2 * X1 + 3

    s1 = np.stack((X1, Y11), axis=0)
    print(s1)
    print(np.cov(s1))
    print(np.cov(X1, Y11))
    print(np.cov(X1, Y12))

    X2 = [1, 1, -1, -1]
    Y2 = [1, -1, -1, 1]
    print(np.cov(X2, Y2))

test3()