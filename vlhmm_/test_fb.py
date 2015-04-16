import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pysam
import vlhmm_.forward_backward as fb
from sklearn.hmm import GaussianHMM
from vlhmm_.context_tr_trie import ContextTransitionTrie
from hmm_.hmm import HMMModel, HMM
from vlhmm_.emission import GaussianEmission, PoissonEmission


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


def sample_(size, n=2, h_states=None, type_emission="Poisson", **e_params):
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

    if len(e_params) > 0:
        emission._set_params(**e_params)
    else:
        emission.set_rand_params()

    for i, state in enumerate(h_states):
        data[i] = emission.sample(state)

    return data, emission


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


def test_hmm(type_e="Poisson", T=int(2e3), start="k-means"):
    def go(vlhmm):
        vlhmm.fit(data, equal_start=start, type_emission=type_e)
        if type_e == "Gauss":
            print("sklearn: {}\nvlhmm: {}".format(sk_log_p, vlhmm._log_p))
        print("real_a", np.exp(model_.log_a))
        name = "graphics/hmm/{}/{}_{}".format(type_e, start, random.randrange(T))
        print(name)
        text = "{}\nT = {}\ninit: {}\n".format(type_e, T, start)
        vlhmm.contexts = ["0", "1"]
        fig = create_img(vlhmm, vlhmm.contexts, log_a, name, text)
        fig.savefig(name+".jpg")

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
    model_ = HMMModel.get_random_model(2, 1)
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


def connect_tries(fname_real_trie, fname_trie, name):
    fig, (ax0, ax1) = plt.subplots(ncols=2)

    ax0.imshow(plt.imread(fname_real_trie))
    ax0.set_title('Real tree')
    ax0.axis('off')

    ax1.imshow(plt.imread(fname_trie))
    ax1.set_title('Predicted tree')
    ax1.axis('off')

    fig.savefig(name)


def create_img(vlhmm, contexts=None, log_a=None, name="", text=""):
    fname_plot = name+"plot_.png"
    vlhmm.plot_log_p().savefig(fname_plot)
    fname_trie = name+"trie_.png"
    vlhmm_contexts, vlhmm_log_a = fb.VLHMMWang\
        .get_sorted_contexts_and_log_a(vlhmm.contexts, vlhmm.log_a, vlhmm.emission.get_order())
    ContextTransitionTrie.draw_context_trie(vlhmm_contexts, vlhmm_log_a, fname_trie)

    if contexts is not None:
        fname_real_trie = name+"real_trie_.png"
        ContextTransitionTrie.draw_context_trie(contexts, log_a, fname_real_trie)

        connect_tries(fname_real_trie, fname_trie, name+"tries.png")
        fig = plt.figure()

        ax = plt.subplot2grid((3, 4), (0, 0), colspan=4, rowspan=2)
        ax.set_title("Log likelihood")
        ax1 = plt.subplot2grid((3, 4), (2, 1))
        ax1.set_title("Real tree")
        ax2 = plt.subplot2grid((3, 4), (2, 2))
        ax2.set_title("Predicted tree")

        ax.imshow(plt.imread(fname_plot))
        ax1.imshow(plt.imread(fname_real_trie))
        ax2.imshow(plt.imread(fname_trie))

        # os.remove(fname_plot)
        # os.remove(fname_trie)
        # os.remove(fname_real_trie)

        for i, ax in enumerate(fig.axes):
                for tl in ax.get_xticklabels() + ax.get_yticklabels():
                    tl.set_visible(False)
        fig.text(0.11, 0.095, text)
        return fig


def main_test(contexts, log_a, T=int(2e3), max_len=4, th_prune=4e-3, log_pr_thresh=0.15,
              type_e="Poisson", start="k-means", save_data=False,show_e=True, **kwargs):
    def go(vlhmm):
        print(type_e)
        vlhmm.fit(data, max_len=max_len, start=start, th_prune=th_prune,
                  log_pr_thresh=log_pr_thresh, type_emission=type_e)
        print(vlhmm.tr_trie.n_contexts, vlhmm.tr_trie.seq_contexts)
        print("T=", T, "max_len=", max_len)
        comp_emission = "real emission\n{}\npredicted emission\n{} \n".format(e_params, vlhmm.emission.get_str_params())
        print(comp_emission)
        path = "graphics/vlhmm2/{}/".format(type_e)
        if not os.path.exists(path):
            os.makedirs(path)
        name = "{}{}{}_{}_{}_{}".format(path, start, random.randrange(T), T, th_prune, max_len)
        print(name)
        text = "{}\nT = {}\ninit: {}\n\n{}\n".format(type_e, T, start, comp_emission)
        fig = create_img(vlhmm, contexts, log_a, name, text)
        fig.savefig(name+".jpg")
        if save_data:
            data_to_file(name+".txt")
        plt.show()

    n= len(log_a)
    h_states = ContextTransitionTrie.sample_(T, contexts, log_a)
    data, emission = sample_(T, n, list(map(int, h_states)), type_emission=type_e, **kwargs)
    e_params  = emission.get_str_params()
    print("real emission:\n{}".format(e_params))
    if show_e:
        emission.show()
        plt.show()
    go(fb.VLHMMWang(n))



if __name__ == "__main__":
    contexts = ["00", "01", "10", "110", "111"]
    log_a = np.log(np.array(
        [[0.8, 0.4, 0.3, 0.2, 0.9],
         [0.2, 0.6, 0.7, 0.8, 0.1]]
    ))

    contexts = ["00", "01", "1"]
    log_a = np.log(np.array(
        [[0.7, 0.4, 0.2],
         [0.3, 0.6, 0.8]]
    ))

    contexts = ["0", "1"]
    log_a = np.log(np.array(
        [[0.8, 0.4],
         [0.2, 0.6]]
    ))
    #
    contexts = ["0", "1", "2"]
    log_a = np.log(np.array(
        [[0.1, 0.3, 0.4],
         [0.3 , 0.1, 0.5],
         [0.6, 0.6, 0.1]]
    ))

    contexts = [""]
    log_a = np.log(np.array(
        [[0.1],
         [0.3],
         [0.6]]
    ))


    contexts = ["00", "01", "02", "1", "2"]
    log_a = np.log(np.array(
        [[0.1, 0.2, 0.3, 0.9, 0.6],
        [0.4, 0.7, 0.1, 0.02, 0.35],
        [0.5, 0.1, 0.6, 0.08, 0.05]]
    ))


    contexts = ["000", "0010", "0011", "01", "1"]
    log_a = np.log(np.array([
        [0.9, 0.2, 0.4, 0.3, 0.9],
        [0.1, 0.8, 0.6, 0.7, 0.1]
    ]
    ))
    #
    # contexts = ["00", "01", "1"]
    # log_a = np.log(np.array([
    #     [0., 0.5, 0.19],
    #     [1., 0.5, 0.81]]))
    # contexts = ["0", "1"]
    # log_a = np.log(np.array(
    #     [[0., 1.],
    #      [1., 0.]]
    # ))

    # contexts = ["00", "01", "10", "110", "111"]
    # log_a = np.log(np.array(
    #     [[0.8, 0.4, 0.3, 0.2, 0.9],
    #      [0.2, 0.6, 0.7, 0.8, 0.1]]
    # ))
    main_test(contexts, log_a, max_len=5, start="k-means", type_e="Poisson", T=int(1e4), th_prune=8e-3, show_e=True)





def get_data(fname="resources/ENCFF000AWF.bam", chr_i=20, bin_size=10000):
        samfile = pysam.AlignmentFile(fname, "rb")
        chr_name = samfile.references[chr_i]

        N = samfile.lengths[chr_i]
        n_bins = int(N/bin_size)+1
        x = np.zeros(n_bins)

        print(chr_name, N)
        print("bin_size = ", bin_size)
        print("n_bins", n_bins)

        for i, read in enumerate(samfile.fetch(chr_name)):
                if not(read.is_unmapped or read.is_duplicate):
                        ind = round(read.pos / bin_size)
                        x[ind] += 1

        np.savetxt("resources/{}_{}.txt".format(chr_name, bin_size), x)
        samfile.close()

        return x


def test_wang_with_data_from_file(f_name, type_e="Poisson", X=None, n=3, max_len=3, th_prune=0.01, log_pr_thresh=0.05, start="k-means", path = "graphics/real/"):
    data = data_from_file(f_name)
    print(len(data))
    vlhmm = fb.VLHMMWang(n)
    print(start)
    vlhmm.fit(data, X=X,  max_len=max_len, start=start,
              th_prune=th_prune, log_pr_thresh=log_pr_thresh, type_emission=type_e)
    print(vlhmm.tr_trie.n_contexts, vlhmm.tr_trie.seq_contexts)
    print("T=", len(data), "max_len=", max_len)


    if not os.path.exists(path):
        os.makedirs(path)
    name = "{}{}_{}_{}".format(path, start, vlhmm.emission.get_str_params(), random.randrange(100))
    print(name)
    create_img(vlhmm, name=name)


# test_wang_with_data_from_file("resources/check_test.txt", n=2, max_len=3, start="k-means", path="graphics/test/check/")
