import random
import numpy as np
import matplotlib.pyplot as plt
from vlhmm_.vlhmm import ContextTrie
from vlhmm_.vlhmm_dumont import ContextTrieDumont


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


def random_string(n, alphabet="abc"):
    return "".join([random.choice(alphabet) for i in range(n)])


def dumont_context_test():
    data = random_string(1450)
    print(data)
    context_trie_dumont = ContextTrieDumont(data, max_len=5)
    print(context_trie_dumont.contexts)
    for i in range(10):
        print(context_trie_dumont.recount_tr_trie(), context_trie_dumont.recount_c_tr_trie())



if __name__ == "__main__":
    dumont_context_test()