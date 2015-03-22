import random
import numpy as np
import matplotlib.pyplot as plt
import vlhmm_.forward_backward as fb
from hmm_.hmm import HMMModel, HMM


def get_data(n, n_components=3):
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


def show_data(X, ax=plt):
    if X.ndim < 2:
        ax.plot(np.linspace(0, 1, len(X)), X, alpha=0.4)
    else:
        ax.plot(X[:, 0], X[:, 1], 'o', alpha=0.4)
    ax.axis('equal')


def data_to_file(data, f_name):
    np.savetxt(f_name, data)


def data_from_file(f_name):
    return np.genfromtxt(f_name)


def test_wang_mixture():

    def go(vlhmm):
        vlhmm.fit(data, max_len=3, n_iter=5)
        print(vlhmm.tr_trie.contexts)
        # print(vlhmm.sample(10))

        fig = plt.figure()
        ax = fig.add_subplot(121)
        show_data(data)

        ax = fig.add_subplot(122)
        # sample_data = vlhmm.sample(3*n*T)
        # show_data(sample_data, ax)
        # plt.show()

    T = 200
    n = 2
    data = get_data(T, n_components=n)
    go(fb.VLHMMWang(n))
    plt.show()


def test_wang_with_hmm_sample():
    def go(vlhmm):
        print(data)
        vlhmm.fit(data, max_len=3, n_iter=5, th_prune=1e-3)
        print(vlhmm.tr_trie.n_contexts, vlhmm.tr_trie.seq_contexts)


    n, m, T = 2, 3, 200
    a = np.array([[0.2, 0.8],
                  [0.6, 0.4]])
    b = np.array([[0.1, 0.6, 0.3],
                  [0.4, 0.2, 0.4]])
    pi = np.array([0.5, 0.5])


    model_ = HMMModel.get_model_from_real_prob(a, b, pi)
    data,_ = model_.sample(T)
    p, model_, h_states = HMM(n).optimal_model(data, m=m, n_starts=3, log_eps=2e-3, max_iter=1e2)
    print(p)
    print(model_)
    print(h_states)

    go(fb.VLHMMWang(n))

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
