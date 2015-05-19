import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pysam
import time
import vlhmm_.forward_backward as fb
from vlhmm_.context_tr_trie import ContextTransitionTrie
from hmm_.hmm import HMMModel
from vlhmm_.emission import GaussianEmission, PoissonEmission
from vlhmm_.poisson_hmm import PoissonHMM


def data_to_file(data, f_name):
    np.savetxt(f_name, data)


def data_from_file(f_name):
    return np.genfromtxt(f_name)


def sample_(size, n=2, h_states=None, type_emission="Poisson", emission=None,
            **e_params):
    if h_states is None:
        model_ = HMMModel.get_random_model(n, n)
        data, h_states = model_.sample(size)
        print("a", np.exp(model_.log_a))

    data = np.zeros(size)

    if emission is None:
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


def poisson_hmm(arr_data, _path, text=""):
    hmm = PoissonHMM(n_components=2)
    logprob, _ = hmm.fit(arr_data)
    with open(_path + "PoissonHMM.txt", "wt") as f:
        f.write("a {}\n\n".format(hmm.transmat_))
        f.write("log_a {}\n\n".format(hmm._log_transmat))
        f.write("lambda: {}\n\n".format(hmm.rates_))
        f.write("\nn_data={}\n".format(len(arr_data)))
        f.write("{}".format(text))
    print()
    return logprob


def connect_tries(fname_real_trie, fname_trie, name, s_real_trie, s_predicted_trie):
    fig, (ax0, ax1) = plt.subplots(ncols=2)

    ax0.imshow(plt.imread(fname_real_trie))
    ax0.set_title(s_real_trie)
    ax0.axis('off')

    ax1.imshow(plt.imread(fname_trie))
    ax1.set_title(s_predicted_trie)
    ax1.axis('off')

    fig.savefig(name)


def create_img(vlhmm, contexts=None, log_a=None, name="", text="",
               language="en"):

    if language == "ru":
        s_log_likelihood = "Логарифм правдоподобия"
        s_real_trie = "Реальное\nдерево"
        s_predicted_trie = "Предсказанное\nдерево"
    else:
        s_log_likelihood = "Log likelihood"
        s_real_trie = "Real tree"
        s_predicted_trie = "Predicted tree"

    fname_plot = name+"plot_.png"
    vlhmm.plot_log_p(language=language).savefig(fname_plot)
    fname_trie = name+"trie_.png"
    ContextTransitionTrie.draw_context_trie(vlhmm.contexts, vlhmm.log_a,
                                            fname_trie)

    if contexts is not None:
        fname_real_trie = name+"real_trie_.png"
        ContextTransitionTrie.draw_context_trie(contexts, log_a,
                                                fname_real_trie)

        connect_tries(fname_real_trie, fname_trie, name+"tries.png",
                      s_real_trie=s_real_trie,
                      s_predicted_trie=s_predicted_trie)
        fig = plt.figure()

        ax = plt.subplot2grid((3, 4), (0, 0), colspan=4, rowspan=2)
        ax.set_title(s_log_likelihood)
        ax1 = plt.subplot2grid((3, 4), (2, 1))
        ax1.set_title(s_real_trie)
        ax2 = plt.subplot2grid((3, 4), (2, 2))
        ax2.set_title(s_predicted_trie)

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


def go_vlhmm(vlhmm, data, contexts, log_a, path="", T=None,
             real_e_params="unknown", max_len=4, show_res=True,
             comp_with_hmm=True, lang_plot='en', **kwargs):
    print(path)
    time_start = time.time()
    vlhmm.fit(data, max_len=max_len, **kwargs)
    fit_time = time.time()-time_start
    if T is None:
        T = len(data)
    print("T=", T, "max_len=", max_len)
    print(vlhmm.tr_trie.n_contexts, vlhmm.tr_trie.seq_contexts)
    print("context_p:", np.exp(vlhmm.log_context_p))
    comp_emission = "real emission\n{}\npredicted emission\n{} \n"\
        .format(real_e_params, vlhmm.emission.get_str_params())
    print(comp_emission)
    print(path)
    text = "{}\nT = {}\ninit: {}\n\n{}\n".format(vlhmm.emission.name, T,
                                                 vlhmm.start, comp_emission)
    fig = create_img(vlhmm, contexts, log_a, path, text, language=lang_plot)
    fig.savefig(path+'main')
    with open("{}info.txt".format(path), "a") as f:
            f.write("\n\n")
            f.write("T={}  max_len={}\n".format(T, max_len))
            f.write("{}\n".format(kwargs))
            for info in vlhmm.info:
                f.write("{}\n".format(info))
    n = len(log_a)
    if comp_with_hmm:
        if type(data) != list:
            logprob = poisson_hmm([data], _path=path)
        else:
            logprob = poisson_hmm(data, _path=path)
        with open("{}info.txt".format(path), "a") as f:
            f.write("lgprob:\nvlhmm = {},  hmm = {}   diff= {}\n".format(vlhmm._log_p, logprob[-1], vlhmm._log_p-logprob[-1]))
            hmm_n_params = n*n + n -1
            hmm_aic = 2*(hmm_n_params-logprob[-1])
            f.write("params: vlhmm={} hmm={}\n".format(vlhmm.get_n_params(), hmm_n_params))
            f.write("aic:\nvlhmm = {},  hmm = {}   diff= {}\n".format(vlhmm.get_aic(), hmm_aic, vlhmm.get_aic()-hmm_aic))
            f.write("fdr, fndr: {}\n\n".format(vlhmm.estimate_fdr_fndr()))
            f.write("fitting time: {}\n".format(fit_time))

        if show_res:
            plt.show()


def main_fb_wang_test(contexts, log_a, T=int(2e3), max_len=4, th_prune=4e-3,
                log_pr_thresh=0.15, _path="graphics/vlhmm2/",
                type_e="Poisson", start="k-means",
                save_data=False, show_e=True, show_res=True,
                data=None, start_params=None, lang_plot='en',
                **kwargs):

    n = len(log_a)
    real_e_params = "unknown"
    if data is None:
        h_states = ContextTransitionTrie.sample_(T, contexts, log_a)
        data, emission = sample_(T, n, list(map(int, h_states)),
                             type_emission=type_e, **kwargs)
        real_e_params = emission.get_str_params()
    print("real emission:\n{}".format(real_e_params))
    path = "{}{}/".format(_path, random.randrange(T))
    if not os.path.exists(path):
            os.makedirs(path)
    if show_e:
        fig = emission.show()
        fig.savefig("{}real_emission.png".format(path))
        plt.show()

    if save_data:
            data_to_file(data, path+".txt")
    go_vlhmm(fb.VLHMMWang(n), data, contexts, log_a, path=path,
             real_e_params=real_e_params, max_len=max_len, start=start,
             th_prune=th_prune,
             log_pr_thresh=log_pr_thresh, type_emission=type_e,
             show_res=show_res,
             lang_plot=lang_plot,
             start_params=start_params)


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


def test_wang_with_data_from_file(f_name="", type_e="Poisson", X=None, n=2,
                                  max_len=3, th_prune=0.01, log_pr_thresh=0.05,
                                  start="k-means", path="graphics/real/"):
    # data = data_from_file(f_name)
    data = np.array(list("010111001000100")).astype(np.float)
    print(len(data))
    vlhmm = fb.VLHMMWang(n)
    print(start)
    vlhmm.fit(data, X=X,  max_len=max_len, start=start,
              th_prune=th_prune, log_pr_thresh=log_pr_thresh,
              type_emission=type_e)
    print(vlhmm.tr_trie.n_contexts, vlhmm.tr_trie.seq_contexts)
    print("T=", len(data), "max_len=", max_len)


    if not os.path.exists(path):
        os.makedirs(path)
    name = "{}{}_{}_{}".format(path, start, vlhmm.emission.get_str_params(),
                               random.randrange(100))
    print(name)
    create_img(vlhmm, name=name)

# test_wang_with_data_from_file(max_len=1)

# test_wang_with_data_from_file(f_name="tests/check_data.txt", n=2, max_len=2, th_prune=6e-3, start="k-means", path="tests/check/")
# test_wang_with_data_from_file("resources/check_test.txt", n=2, max_len=3, start="k-means", path="graphics/test/check/")

# X = "000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000001111000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001101000000000000000000"
# X = X.replace("0", "2").replace("1","0").replace("2", "1")
# test_wang_with_data_from_file("resources/check_test.txt", n=2, max_len=3, start="k-means", path="graphics/test/check/")
# test_wang_with_data_from_file("resources/check_test_2_.txt", n=2, max_len=3, start="k-means", path="graphics/test/check2/")
# test_wang_with_data_from_file("resources/chr21_10000.txt", max_len=4, start="k-means", type_e="Poisson")


def get_real_data(chr_i=1, bin_size=400, thr=10):
    data = np.genfromtxt("resources/chr{}_{}.txt".format(chr_i, bin_size))
    arr_data = []
    t_start, t_fin = 0, 1
    for i, x in enumerate(data):
        if x == 0:
            if t_fin - t_start > thr:
                arr_data.append(np.array(data[t_start: t_fin]))
            t_start = t_fin
        t_fin += 1
    return arr_data


def independent_fitting_parts(chr_i, bin_size, n=2, max_len=3, start="k-means"):
    def fit_part(Y_):
        Y = np.array(Y_)
        print("T = {}".format(len(Y)))
        print(Y)
        vlhmm = fb.VLHMMWang(n)
        vlhmm.fit(np.array(Y), max_len=max_len, start=start)
        name = "{}{}_{}_{}_{}".format(path, start, len(Y),
                                      vlhmm.emission.get_str_params(), ind)
        print(name)
        create_img(vlhmm, name=name)

    path = "graphics/real/chr{}/{}/{}/".format(chr_i, bin_size,
                                               random.randrange(1000))
    print(path)
    if not os.path.exists(path):
        os.makedirs(path)

    thr = 110
    arr_data = get_real_data(chr_i, bin_size, thr)
    for ind, data in enumerate(arr_data):
        fit_part(data)

def go_main_fb_wang_test():
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
        [0.9, 0.2, 0.65, 0.3, 0.9],
        [0.1, 0.8, 0.35, 0.7, 0.1]
    ]
    ))

    contexts = ["00", "01", "1"]
    log_a = np.log(np.array([
        [0., 0.5, 0.19],
        [1., 0.5, 0.81]]))
    # contexts = ["0", "1"]
    # log_a = np.log(np.array(
    #     [[0., 1.],
    #      [1., 0.]]
    # ))

    contexts = ["0", "1"]
    log_a = np.log(np.array(
        [[0.8, 0.4],
         [0.2, 0.6]]
    ))

    contexts = ["00", "01", "1"]
    log_a = np.log(np.array(
        [[0.7, 0.4, 0.2],
         [0.3, 0.6, 0.8]]
    ))
    # contexts = [""]
    # log_a = np.log(np.array(
    #     [[0.4],
    #      [0.6]]
    # ))

    # contexts = ["000", "0010", "0011", "01", "1"]
    # log_a = np.log(np.array([
    #     [0.9, 0.2, 0.65, 0.3, 0.9],
    #     [0.1, 0.8, 0.35, 0.7, 0.1]
    # ]
    # ))
    #
    #
    contexts = ["00", "01", "10", "110", "111"]
    log_a = np.log(np.array(
        [[0.8, 0.4, 0.3, 0.2, 0.9],
         [0.2, 0.6, 0.7, 0.8, 0.1]]
    ))
    # contexts = ["0", "1"]
    # log_a = np.log(np.array(
    #     [[0.8, 0.4],
    #      [0.2, 0.6]]
    # ))
    # contexts = [""]
    # log_a = np.log(np.array(
    #     [[0.4],
    #      [0.6]]
    # ))

    # contexts = ["00", "010", "011", "1"]
    # log_a = np.log(np.array(
    #     [[0.7, 0.6, 0.2, 0.4],
    #      [0.3, 0.4, 0.8, 0.6]]
    # ))

    # contexts = ['0', '1']
    # log_a = np.log(
    # [[0.2,   0.6],
    #  [0.8,   0.4]])
    log_c_p = np.ones(len(contexts))/len(contexts)
    alpha = [2., 23.1]
    start_params = dict(log_a=log_a, contexts=contexts, log_c_p=log_c_p,
                        alpha=alpha)
    start_params = None
    data = data_from_file("tests/data")
    main_fb_wang_test(contexts, log_a, max_len=4, start="k-means",
                      type_e="Poisson", T=int(4e3), th_prune=0.01, show_e=False,
                      data=data,
                      _path="ru/vlhmm/",
                      log_pr_thresh=0.01, alpha=alpha,
                      start_params=start_params,
                      lang_plot="ru")

def go_independent_parts_test(ch):
    chr_i = 1
    bin_size = 400
    independent_fitting_parts(chr_i=chr_i, bin_size=bin_size,
                              max_len=3, start="k-means")

if __name__ == "__main__":
    go_main_fb_wang_test()

