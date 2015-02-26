import random
import numpy as np
import matplotlib.pyplot as plt
from vlmm_ import vlhmm_wang


def get_data(n, n_components=3):
    X = np.zeros((n_components * n, 2))
    _var = 10.
    for i in range(n_components):
        mean = ([random.randrange(_var / 2),
                 random.randrange(_var / 2)] + np.random.random((2,))) * _var
        cov = np.random.random((2, 2)) * _var
        x, y = np.random.multivariate_normal(mean, cov, n).T
        X[n * i:n * (i + 1)] = np.c_[x, y]
    return np.random.permutation(X)


def show_data(X):
    plt.plot(X[:, 0], X[:, 1], 'x')
    plt.axis('equal')
    plt.show()


def test():
    data = get_data(200)
    # show_data(data)
    n = 2
    vlhmm = vlhmm_wang.VLHMM(n)
    vlhmm.fit(data, max_len=3, n_iter=3)


test()
