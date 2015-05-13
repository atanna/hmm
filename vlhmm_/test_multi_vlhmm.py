import os
import random
import numpy as np
import pylab as plt
from scipy.stats.mstats_basic import mquantiles
from vlhmm_.context_tr_trie import ContextTransitionTrie
from vlhmm_.emission import PoissonEmission
from vlhmm_.poisson_hmm import PoissonHMM
from vlhmm_.test_fb import create_img, data_to_file, sample_, go_vlhmm
from vlhmm_.multi_vlhmm import MultiVLHMM
from warnings import filterwarnings



def get_parts(data, boards_parts=[0., 0.5, 0.75, 1.]):
    T = len(data)

    t = mquantiles(list(range(T + 1)), boards_parts)
    print(t)
    N = len(t)
    return [data[t[i - 1]:t[i]] for i in range(1, N)]


def main_multi_vlhmm_test(contexts, log_a, T=int(1e3), arr_T=None,
                          arr_data=None, max_len=4,
                          th_prune=9e-3, log_pr_thresh=0.01, n_parts=10,
                          max_log_p_diff=1.5, type_e="Poisson",
                          start="k-means",
                          save_data=False, show_e=True,
                          _path="graphics/multi/sample/",
                          start_params=None, **e_params):
    n = len(log_a)
    real_e_params = "unknown"
    if arr_data is None:
        if arr_T is None:
            arr_T = [int(T/n_parts)] * n_parts
        emission = PoissonEmission(n_states=n)
        emission._set_params(**e_params)
        arr_data, emission = zip(
            *[sample_(T, n, list(map(int, ContextTransitionTrie
                                     .sample_(T, contexts, log_a))),
                      emission=emission)
              for T in arr_T])
        emission = emission[0]
        T = sum(len(data) for data in arr_data)
        real_e_params = emission.get_str_params()
        print("real emission:\n{}".format(real_e_params))
        if show_e:
            emission.show()
            plt.show()

    path = "{}/{}/".format(_path, random.randrange(T))
    if not os.path.exists(path):
        os.makedirs(path)
    if save_data:
        data_to_file(arr_data, path + "multi_data.txt")

    with open("{}info.txt".format(path), "wt") as f:
        f.write("T={}\nstart={}\nmax_len={}\nth_prune={}\n"
                "log_pr_thresh={}\narr_T={}\n"
                .format(T, start, max_len, th_prune,
                        log_pr_thresh, arr_T))
        for d in arr_data:
            f.write("{}".format(d))
    T = [len(d) for d in arr_data]

    vlhmm = MultiVLHMM(n)
    go_vlhmm(vlhmm, list(arr_data), contexts, log_a, path=path, T=T,
             real_e_params=real_e_params, max_len=max_len, start=start,
             th_prune=th_prune,
             log_pr_thresh=log_pr_thresh, type_emission=type_e,
             max_log_p_diff=max_log_p_diff,
             start_params=start_params)


def get_real_data(chr_i=1, bin_size=400, thr=10, max_num=-1):
    data = np.genfromtxt("resources/chr{}_{}.txt".format(chr_i, bin_size))
    arr_data = []
    t_start, t_fin = 0, 1
    sample_positions = []
    for i, x in enumerate(data):
        if x == 0:
            if t_fin - t_start > thr:
                arr_data.append(np.array(data[t_start: t_fin]))
                sample_positions\
                    .append(np.array(list(range(t_start, t_fin)))
                            .astype(np.int)*bin_size)
            t_start = t_fin
        t_fin += 1
    _T = len(arr_data)

    def sort_sample(arr_data, sample_positions):
        arr_data, sample_positions = zip(*sorted(zip(arr_data, sample_positions),
                      key=lambda x:len(x[0]), reverse=True))
        return np.array(arr_data), np.array(sample_positions)
    arr_data, sample_positions = sort_sample(arr_data, sample_positions)
    T = max_num if max_num > 0 else _T
    tail_T = int(0.6*T)
    perm = np.random.permutation(_T-tail_T)
    arr_data[tail_T:] = np.array(arr_data[tail_T:])[perm]
    sample_positions[tail_T:] = np.array(sample_positions[tail_T:])[perm]
    return sort_sample(arr_data[:T], sample_positions[:T])


def real_test(arr_data, max_len=4, th_prune=6e-3, log_pr_thresh=0.01,
              type_e="Poisson", start="k-means", n=2,
              _path="graphics/multi/real/", write_data=True,
              comp_with_hmm=True,
              sample_pos=None,
              **kwargs):
    def go(vlhmm):
        name = path
        vlhmm.fit(arr_data, max_len=max_len, start=start, th_prune=th_prune,
                  log_pr_thresh=log_pr_thresh, type_emission=type_e, **kwargs)
        print(vlhmm.tr_trie.n_contexts, vlhmm.tr_trie.seq_contexts)
        print("T=", T, "max_len=", max_len)
        print(vlhmm.emission.get_str_params())
        print(path)
        print("aic:", vlhmm.get_aic())
        create_img(vlhmm, name=name)
        with open("{}info.txt".format(path), "a") as f:
            # f.write("\nemission: {}\n".format(vlhmm.emission.get_str_params()))
            # f.write("contexts: {}\n".format(vlhmm.contexts))
            # f.write("a: {}\n".format(np.round(np.exp(vlhmm.log_a), 4)))
            # f.write("log_c_p: {}\n".format(vlhmm.log_tcontext_p))
            # f.write("c_p: {}\n\n".format(np.exp(vlhmm.log_context_p)))
            for info in vlhmm.info:
                f.write("{}\n".format(info))

        print(comp_with_hmm)
        if comp_with_hmm:
            logprob = poisson_hmm(arr_data, _path=path)
            with open("{}info.txt".format(path), "a") as f:
                f.write("lgprob:\nvlhmm = {},  hmm = {}   diff= {}\n".format(vlhmm._log_p, logprob[-1], vlhmm._log_p-logprob[-1]))
                hmm_n_params = n*(n-1) + n + (n-1)
                hmm_aic = 2*(hmm_n_params-logprob[-1])
                f.write("params: vlhmm={} hmm={}\n".format(vlhmm.get_n_params(), hmm_n_params))
                f.write("aic:\nvlhmm = {},  hmm = {}   diff= {}\n".format(vlhmm.get_aic(), hmm_aic, vlhmm.get_aic()-hmm_aic))
                f.write("fdr, fndr: {}\n".format(vlhmm.estimate_fdr_fndr()))

        with open("{}peaks.txt".format(path), "wt") as f:
            states = vlhmm.get_hidden_states()
            f.write("\npeaks_pos:\n".format())
            for i, st in enumerate(states):
                peaks = sample_pos[i][st==1]
                if len(peaks) > 0:
                    f.write("{}:  len_sample={},  {}%\n{}\n\n".format(i, len(st), float(len(peaks))/len(st),
                                                  peaks))


    path = "{}/{}_{}_{}/".format(_path, max_len, start, random.randrange(1e3))
    print(start)
    if not os.path.exists(path):
        os.makedirs(path)
    print(path)
    T = sum(len(d) for d in arr_data)
    with open("{}info.txt".format(path), "wt") as f:
        f.write("T={}\nstart={}\nmax_len={}\nth_prune={}\nlog_pr_thresh={}\n"
                .format(T, start, max_len, th_prune, log_pr_thresh))
        f.write(
            "T {} {}\n\n".format(len(arr_data), [len(d) for d in arr_data]))
        if write_data:
            for i, d in enumerate(arr_data):
                f.write("{}:\n{}\n".format(i, d))
                if i > 10:
                    f.write("...\n")
                    break
        f.write("\n\n")

    go(MultiVLHMM(n))


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


def go_sample_test():
    arr_T = None
    alpha = None

    contexts = ["0", "1"]
    log_a = np.log(np.array(
        [[0.8, 0.4],
         [0.2, 0.6]]
    ))


    contexts = ["00", "01", "10", "110", "111"]
    log_a = np.log(np.array(
        [[0.8, 0.4, 0.3, 0.2, 0.9],
         [0.2, 0.6, 0.7, 0.8, 0.1]]
    ))

    # alpha = np.array([5.4,  40.3])
    # contexts = ['00', '010', '011', '1']
    # log_a = np.log(np.array(
    #     [[0.9462,  0.5248,  1., 0.7132],
    #      [0.0538,  0.4752,  0., 0.2868]]))
    # arr_T = [51, 51, 61, 52, 65, 58, 69]
    T = int(4e3)
    n_parts = 8
    alpha=[2.,   23.1]
    # contexts = [""]
    # log_a = np.log(np.array(
    #     [[0.4],
    #      [0.6]]
    # ))
    contexts = ["00", "01", "1"]
    log_a = np.log(np.array(
        [[0.7, 0.4, 0.2],
         [0.3, 0.6, 0.8]]
    ))

    # contexts = ["00", "010", "011", "1"]
    # log_a = np.log(np.array(
    #     [[0.7, 0.6, 0.2, 0.4],
    #      [0.3, 0.4, 0.8, 0.6]]
    # ))

    # contexts = ["00", "01", "100", "101", "11"]
    # log_a = np.log(np.array(
    #     [[0.8, 0.4, 0.3, 0.9,  0.2],
    #      [0.2, 0.6, 0.7, 0.1, 0.8]]
    # ))

    log_c_p = np.ones(len(contexts))/len(contexts)
    alpha = [2., 19]
    start_params = dict(log_a=log_a, contexts=contexts, log_c_p=log_c_p,
                        alpha=alpha)
    start_params=None

    main_multi_vlhmm_test(contexts, log_a, T=T, arr_T=arr_T, max_len=4,
                          log_pr_thresh=0.05,
                          n_parts=n_parts, th_prune=0.007, start="k-means",
                          show_e=False, _path="graphics/multi/special_sample/",
                          alpha=alpha, start_params=start_params)


def go_real_test():
    for chr_i in range(4, 5):
        bin_size = 200
        max_len = 4
        thr = 5
        max_num = 12000
        arr_data, sample_pos = get_real_data(chr_i, bin_size, thr=thr,
                                             max_num=max_num)
        print(len(arr_data))
        try:
            start_params=None
            # contexts= ['0000', '0001', '001', '01', '1']
            # log_c_p = np.log([0.975, 0.004, 0.004, 0.004, 0.013])
            # log_a = np.log([[1., 0.95, 0.91, 0.77, 0.3],
            #                [0., 0.05, 0.09, 0.23, 0.7]])
            # alpha = [2., 23.1]
            # start_params = dict(log_a=log_a, contexts=contexts, log_c_p=log_c_p,
            #             alpha=alpha)
            real_test(arr_data,
                      _path="graphics_test/multi/real/chr_{}/bin_size_{}/min_len_seq_{}".format(
                          chr_i, bin_size, thr),
                      max_len=max_len, start="k-means", log_pr_thresh=0.5,
                      th_prune=0.004,
                      start_params=start_params,
                      sample_pos=sample_pos)
        except Exception as e:
            print(e)
            continue



if __name__ == "__main__":
    filterwarnings("ignore")
    # go_sample_test()
    go_real_test()





