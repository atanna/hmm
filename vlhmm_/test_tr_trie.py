import random
import datrie
import numpy as np
import matplotlib.pyplot as plt
from vlhmm_.context_tr_trie import ContextTransitionTrie


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


def print_all_p(items):
        for s, val in items:
            print("p({}|{}) = {}".format(s[0], s[1:], np.exp(val)))



def test_context_tr_trie_with_data():

    data = random_string(100, "10")
    print(data)
    tr_prune = 1e-3
    context_transition_trie = ContextTransitionTrie(data)
    print(context_transition_trie.seq_contexts)
    context_transition_trie.prune(tr_prune)
    print("prune", context_transition_trie.seq_contexts)
    print(context_transition_trie.log_c_tr_trie.items())


# test_context_tr_trie_with_data()


def test_context_tr_trie():
    alphabet = "01"
    contexts = ["01","00","11","10"]
    log_a = np.log(np.array(
        [[0.3,0.3, 0.6, 0.58],
         [0.7,0.7, 0.4, 0.42]]))
    th_prune = 1e-3
    c_tr_trie = ContextTransitionTrie(alphabet)
    c_tr_trie.recount_with_log_a(log_a, contexts)
    print("contexts:", c_tr_trie.seq_contexts)

    c_tr_trie.prune(th_prune)

    print("prune", c_tr_trie.seq_contexts)
    print_all_p(c_tr_trie.log_c_tr_trie.items())



test_context_tr_trie()



