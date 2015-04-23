import os
import random
import numpy as np
import pylab as plt
from scipy.stats.mstats_basic import mquantiles
from vlhmm_.context_tr_trie import ContextTransitionTrie
from vlhmm_.test_fb import create_img, data_to_file, sample_, go_vlhmm
from vlhmm_.multi_vlhmm import MultiVLHMM
from vlhmm_.tests.poisson_hmm import PoissonHMM


def get_parts(data, boards_parts=[0., 0.5, 0.75, 1.]):
    T = len(data)

    t = mquantiles(list(range(T+1)), boards_parts)
    print(t)
    N = len(t)
    return [data[t[i-1]:t[i]] for i in range(1, N)]


def main_multi_vlhmm_test(contexts, log_a, T=int(1e3), arr_data=None, max_len=4,
                th_prune=9e-3, log_pr_thresh=0.01, n_parts=10,
                max_log_p_diff=1.5, type_e="Poisson", start="k-means",
                save_data=False, show_e=True,
                _path="graphics/multi/sample/", **kwargs):

    n = len(log_a)
    h_states = ContextTransitionTrie.sample_(T, contexts, log_a)
    real_e_params = "unknown"
    boards_parts = None
    if arr_data is None:
        arr_data, emission = sample_(T, n, list(map(int, h_states)),
                                     type_emission=type_e, **kwargs)
        real_e_params = emission.get_str_params()
        print("real emission:\n{}".format(real_e_params))
        if show_e:
            emission.show()
            plt.show()
        T = len(arr_data)
        boards_parts = np.array(list(range(n_parts+1)))/n_parts
        arr_data = get_parts(arr_data, boards_parts)

    path = "{}/{}/".format(_path, random.randrange(T))
    if not os.path.exists(path):
            os.makedirs(path)
    print(path)
    if save_data:
            data_to_file(arr_data, path+"multi_data.txt")

    with open("{}info.txt".format(path), "wt") as f:
        f.write("T={}\nstart={}\nmax_len={}\nth_prune={}\n"
                "log_pr_thresh={}\nboards_parts={}\n"
                .format(T, start, max_len, th_prune,
                        log_pr_thresh, boards_parts))
        f.write("T {}\n\n".format([len(d) for d in arr_data]))
        for d in arr_data:
            f.write("{}".format(d))
    T = [len(d) for d in arr_data]

    go_vlhmm(MultiVLHMM(n), arr_data, contexts, log_a, name=path, T=T,
             real_e_params=real_e_params, max_len=max_len, start=start,
             th_prune=th_prune,
             log_pr_thresh=log_pr_thresh, type_emission=type_e,
             max_log_p_diff=max_log_p_diff)


def get_real_data(chr_i=1, bin_size=400, thr=10):
    data = np.genfromtxt("../resources/chr{}_{}.txt".format(chr_i, bin_size))
    arr_data = []
    t_start, t_fin = 0, 1
    for i, x in enumerate(data):
        if x == 0:
            if t_fin - t_start > thr:
                arr_data.append(np.array(data[t_start: t_fin]))
            t_start = t_fin
        t_fin += 1
    return arr_data


def real_test(arr_data, max_len=4, th_prune=6e-3, log_pr_thresh=0.01,
              type_e="Poisson", start="k-means", n=2,
              _path="graphics/multi/real/", **kwargs):
    def go(vlhmm):
        name = path
        vlhmm.fit(arr_data, max_len=max_len, start=start, th_prune=th_prune,
                  log_pr_thresh=log_pr_thresh, type_emission=type_e, **kwargs)
        print(vlhmm.tr_trie.n_contexts, vlhmm.tr_trie.seq_contexts)
        print("T=", T, "max_len=", max_len)
        print(vlhmm.emission.get_str_params())
        print(path)

        create_img(vlhmm, name=name)
        with open("{}info.txt".format(path), "a") as f:
            f.write("\nemission: {}\n".format(vlhmm.emission.get_str_params()))
            f.write("contexts: {}\n".format(vlhmm.contexts))
            f.write("a: {}\n".format(np.round(np.exp(vlhmm.log_a), 4)))
            f.write("log_c_p: {}\n".format(vlhmm.log_context_p))
            f.write("c_p: {}\n".format(np.exp(vlhmm.log_context_p)))
        # plt.show()

    path = "{}/{}_{}_{}/".format(_path, max_len, start, random.randrange(1e3))
    print(start)
    if not os.path.exists(path):
            os.makedirs(path)
    print(path)
    T = sum(len(d) for d in arr_data)
    with open("{}info.txt".format(path), "wt") as f:
        f.write("T={}\nstart={}\nmax_len={}\nth_prune={}\nlog_pr_thresh={}\n"
              .format(T, start, max_len, th_prune,log_pr_thresh))
        f.write("T {} {}\n\n".format(len(arr_data), [len(d) for d in arr_data]))
        if len(arr_data) < 100:
            for i, d in enumerate(arr_data):
                f.write("{}:\n{}\n".format(i, d))

    go(MultiVLHMM(n))


def poisson_hmm(arr_data, _path, thr):
    hmm = PoissonHMM(n_components=2)
    hmm.fit(arr_data)
    with open(_path+"PoissonHMM.txt", "wt") as f:
        f.write("a {}\n\n".format(hmm.transmat_))
        f.write("log_a {}\n\n".format(hmm._log_transmat))
        f.write("lambda: {}\n\n".format(hmm.rates_))
        f.write("\nthr={}\nn_data={}".format(thr, len(arr_data)))
    print()


def go_sample_test():
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

    # contexts = ["00", "01", "10", "110", "111"]
    # log_a = np.log(np.array(
    #     [[0.8, 0.4, 0.3, 0.2, 0.9],
    #      [0.2, 0.6, 0.7, 0.8, 0.1]]
    # ))

    main_multi_vlhmm_test(contexts, log_a, T=int(1e3), max_len=2,
              max_log_p_diff=1.5,
              n_parts=5, th_prune=6e-3, start="k-means", show_e=False)


def go_real_test():
    for chr_i in range(1, 22):
        bin_size = 200
        max_len = 4
        thr = 45

        arr_data = get_real_data(chr_i, bin_size, thr=thr)
        print(len(arr_data))
        try:
            real_test(arr_data,
                      _path="graphics2/multi/real/chr_{}/bin_size_{}/min_len_seq_{}".format(chr_i, bin_size, thr),
                      max_len=max_len, start="k-means", max_log_p_diff=1.5)
        except Exception:
            continue


    # poisson_hmm(arr_data, _path="graphics/multi/real/chr_{}/{}/"
    #             .format(chr_i, bin_size), thr=thr)


if __name__ == "__main__":
    # go_sample_test()
    go_real_test()