import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.hmm import GaussianHMM
import vlhmm_.forward_backward as fb
from hmm_.hmm import HMMModel, HMM
from vlhmm_.vlhmm import GaussianEmission


def data_to_file(data, f_name):
    np.savetxt(f_name, data)


def data_from_file(f_name):
    return np.genfromtxt(f_name)


def get_mixture(n, n_components=3):
    X = np.zeros((n_components * n, 2))
    _var = 10.
    for i in range(n_components):
        mean = ([random.randrange(_var / 2),
                 random.randrange(_var / 2)] + np.random.random((2,))) * _var
        cov = np.random.random((2, 2)) * _var
        print("mean, cov", mean, cov)
        x, y = np.random.multivariate_normal(mean, cov, n).T
        X[n * i:n * (i + 1)] = np.c_[x, y]
    return np.random.permutation(X)


def rand_params(n):
    mu = []
    sigma = []
    _var = 10.
    for i in range(n):
        mu.append(([random.randrange(_var / 2),
                 random.randrange(_var / 2)] + np.random.random((2,))) * _var)
        sigma.append(np.random.random((2, 2)) * _var)
    return np.array(mu), np.array(sigma)


def sample_hmm(size, n=2, h_states=None):
    if h_states is None:
        model_ = HMMModel.get_random_model(n, n)
        data, h_states = model_.sample(size)
        print("a", np.exp(model_.log_a))

    emission = GaussianEmission()
    emission._set_params(*rand_params(n))
    print("mu", emission.mu)

    data = np.zeros((size, 2))
    for i, state in enumerate(h_states):
        data[i] = emission.sample(state)

    return data


def test_sklearn(data, n=2, n_iter=5):
    hmm = GaussianHMM(n_components=n, n_iter=n_iter)
    hmm.fit([data])
    print("sklearn GaussianHMM:")
    print("a", np.exp(hmm._log_transmat))
    print("mu", hmm.means_)
    log_p =hmm.score(data)
    print("score", log_p)
    return log_p


def test_wang_mixture():

    def go(vlhmm):
        vlhmm.fit(data, max_len=3, n_iter=5)
        print(vlhmm.tr_trie.contexts)

    T = 200
    n = 2
    data = get_mixture(T, n_components=n)
    go(fb.VLHMMWang(n))
    plt.show()


def test_wang_with_hmm_sample():
    def go(vlhmm):
        vlhmm.fit(data, max_len=3, n_iter=15, th_prune=4e-2)
        print(vlhmm.tr_trie.n_contexts, vlhmm.tr_trie.seq_contexts)
        print("sklearn: {}\nvlhmm: {}".format(sk_log_p, vlhmm._log_p))
        print("T=", T)


    n, m, T = 2, 2, int(4e2)
    a = np.array([[0.2, 0.8],
                  [0.6, 0.4]])
    b = np.array([[0.1, 0.9],
                  [0.2, 0.8]])


    model_ = HMMModel.get_model_from_real_prob(a, b)
    data, h_states = model_.sample(T)
    data = sample_hmm(T, n, h_states)

    sk_log_p = test_sklearn(data, n)
    go(fb.VLHMMWang(n))


def test_gauss_hmm():
    def go(vlhmm):
        vlhmm.fit(data, n_iter=n_iter)
        print("sklearn: {}\nvlhmm: {}".format(sk_log_p, vlhmm._log_p))
        print("real_a", np.exp(model_.log_a))
        print(T)

    n, m, T = 2, 2, int(1e3)
    a = np.array([[0.2, 0.8],
                  [0.6, 0.4]])
    b = np.array([[0.1, 0.9],
                  [0.2, 0.8]])


    model_ = HMMModel.get_model_from_real_prob(a, b)
    data, h_states = model_.sample(T)
    data = sample_hmm(T, n, h_states)
    n_iter=100
    sk_log_p = test_sklearn(data, n, n_iter)
    go(fb.HMM_Gauss(n))

    pass


def test_hmm():
    n, m, T = 3, 2, int(3e2)
    n_iter = 8
    model = HMMModel.get_random_model(n, m)
    print(model)
    data, h_states = model.sample(T)

    hmm = fb.HMM(n).fit(data, n_iter+1)
    print(hmm.model)

    print(HMM(n).observation_log_probability(hmm.model, data))
    print()

    log_p, optimal_model, h_opt_states = HMM(n).optimal_model(data, m=m, max_iter=n_iter)
    print("hmm:\n{}\n{}".format(optimal_model, log_p))


def test_wang_with_data_from_file(f_name):
    data = data_from_file(f_name)
    n=3
    vlhmm = fb.VLHMMWang(n)
    vlhmm.fit(data[:,np.newaxis], max_len=3, n_iter=3)


if __name__ == "__main__":
    test_wang_with_hmm_sample()
    # test_wang_mixture()
    # test_gauss_hmm()
