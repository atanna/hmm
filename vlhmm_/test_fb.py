import os
import random
import numpy as np
import matplotlib.pyplot as plt
import vlhmm_.forward_backward as fb
from sklearn.hmm import GaussianHMM
from vlhmm_.context_tr_trie import ContextTransitionTrie
from hmm_.hmm import HMMModel, HMM
from vlhmm_.vlhmm import GaussianEmission, PoissonEmission


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


def sample_(size, n=2, h_states=None, type_emission="Poisson"):
    if h_states is None:
        model_ = HMMModel.get_random_model(n, n)
        data, h_states = model_.sample(size)
        print("a", np.exp(model_.log_a))

    if type_emission == "Poisson":
        emission = PoissonEmission(n_states=n)
        data = np.zeros(size)
    else:
        emission = GaussianEmission(n_states=n)
        data = np.zeros((size, 2))

    emission.set_rand_params()

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
        vlhmm.fit(data, max_len=5, n_iter=15, th_prune=4e-2)
        print(vlhmm.tr_trie.n_contexts, vlhmm.tr_trie.seq_contexts)
        print("sklearn: {}\nvlhmm: {}".format(sk_log_p, vlhmm._log_p))
        print("T=", T)


    n, m, T = 2, 2, int(3e3)
    a = np.array([[0.2, 0.8],
                  [0.6, 0.4]])
    b = np.array([[0.1, 0.9],
                  [0.2, 0.8]])


    model_ = HMMModel.get_model_from_real_prob(a, b)
    data, h_states = model_.sample(T)
    data = sample_(T, n, h_states)

    sk_log_p = test_sklearn(data, n)
    go(fb.VLHMMWang(n))


def test_hmm(type_e="Poisson", T=int(2e3), equal_start=False):
    def go(vlhmm):
        vlhmm.fit(data, equal_start=equal_start, type_emission=type_e)
        if type_e == "Gauss":
            print("sklearn: {}\nvlhmm: {}".format(sk_log_p, vlhmm._log_p))
        print("real_a", np.exp(model_.log_a))
        fig = vlhmm.plot_log_p()
        eq_st = "eq_" if equal_start else ""
        name = "graphics/hmm/{}/{}{}_{}.jpg"\
            .format(type_e, eq_st, T,random.randrange(T))
        fig.text(0.2,0.1, "{}\nstarts: {}\nT={}\nreal_a:\n{}\na:\n{}"
                 .format(type_e,
                         "eq" if equal_start else "k-means",
                         T, np.exp(model_.log_a).T,
                         np.around(np.exp(vlhmm.log_a), 3)))
        plt.show()
        print(type_e)
        print(T)
        print(name)
        fig.savefig(name)
        plt.show()


    n, m = 2, 2
    a = np.array([[0.2, 0.8],
                  [0.6, 0.4]])
    b = np.array([[0.1, 0.9],
                  [0.2, 0.8]])


    model_ = HMMModel.get_model_from_real_prob(a, b)
    _, h_states = model_.sample(T)

    data = sample_(T, n, h_states, type_emission=type_e)
    print("data:", data)

    if type_e == "Gauss":
        sk_log_p = test_sklearn(data, n, 100)

    go(fb.HMM(n))


def test_discrete_hmm():
    n, m, T = 3, 2, int(3e2)
    n_iter = 8
    model = HMMModel.get_random_model(n, m)
    print(model)
    data, h_states = model.sample(T)

    hmm = fb.DiscreteHMM(n).fit(data, n_iter+1)
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


def create_img(vlhmm, contexts, log_a, name, text):
    fig = plt.figure()

    ax = plt.subplot2grid((3, 4), (0, 0), colspan=4, rowspan=2)
    ax.set_title("Log likelihood")
    ax1 = plt.subplot2grid((3, 4), (2, 1))
    ax1.set_title("Real tree")
    ax2 = plt.subplot2grid((3, 4), (2, 2))
    ax2.set_title("Predicted tree")

    fname_plot = name+"plot_.png"
    vlhmm.plot_log_p().savefig(fname_plot)
    fname_real_trie = name+"real_trie_.png"
    ContextTransitionTrie.draw_context_trie(contexts, log_a, fname_real_trie)
    fname_trie = name+"trie_.png"
    ContextTransitionTrie.draw_context_trie(vlhmm.contexts, vlhmm.log_a, fname_trie)


    ax.imshow(plt.imread(fname_plot))
    ax1.imshow(plt.imread(fname_real_trie))
    ax2.imshow(plt.imread(fname_trie))

    # os.remove(fname_plot)
    # os.remove(fname_trie)
    # os.remove(fname_real_trie)

    for i, ax in enumerate(fig.axes):
            for tl in ax.get_xticklabels() + ax.get_yticklabels():
                tl.set_visible(False)
    fig.text(0.20, 0.095, text)
    return fig


def main_test(contexts, log_a, n=2, T=int(2e3), max_len=4,
              type_e="Poisson", equal_start=False):
    def go(vlhmm):
        vlhmm.fit(data, max_len=max_len, equal_start=equal_start,
                  th_prune=4e-3, type_emission=type_e)
        eq_st = "eq_" if equal_start else ""
        print(vlhmm.tr_trie.n_contexts, vlhmm.tr_trie.seq_contexts)
        print("T=", T, "max_len=", max_len)

        name = "graphics/vlhmm/{}/{}{}".format(type_e, eq_st, random.randrange(T))
        print(name)
        text = "{}\nT = {}\ninit: {}\n".format(type_e, T,
                         "equal" if equal_start else "k-means")
        fig = create_img(vlhmm, contexts, log_a, name, text)
        fig.savefig(name+".jpg")
        plt.show()

    h_states = ContextTransitionTrie.sample_(T, contexts, log_a)
    data = sample_(T, n, list(map(int, h_states)), type_emission=type_e)
    go(fb.VLHMMWang(n))



if __name__ == "__main__":
    contexts = ["00", "01", "10", "110", "111"]
    log_a = np.log(np.array(
        [[0.8, 0.4, 0.3, 0.2, 0.9],
         [0.2, 0.6, 0.7, 0.8, 0.1]]
    ))

    contexts = ["00", "01", "1"]
    log_a = np.log(np.array(
        [[0.7, 0.4, 0.3],
         [0.3, 0.6, 0.7]]
    ))

    contexts = ["0", "1"]
    log_a = np.log(np.array(
        [[0.8, 0.4],
         [0.2, 0.6]]
    ))

    # contexts = [""]
    # log_a = np.log(np.array(
    #     [[0.4],
    #      [0.6]]
    # ))

    main_test(contexts, log_a, max_len=4, equal_start=False, type_e="Gauss", T=int(4e3))
