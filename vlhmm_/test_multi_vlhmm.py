import os
import random
import numpy as np
import pylab as plt
from scipy.stats.mstats_basic import mquantiles
from vlhmm_.context_tr_trie import ContextTransitionTrie
from vlhmm_.test_fb import create_img, data_to_file, sample_, data_from_file
from vlhmm_.multi_vlhmm import MultiVLHMM


def get_parts(data, boards_parts=[0., 0.5, 0.75, 1.]):
    T = len(data)

    t = mquantiles(list(range(T+1)), boards_parts)
    print(t)
    N = len(t)
    return [data[t[i-1]:t[i]] for i in range(1, N)]


def main_test(contexts, log_a, T=int(1e3), data=None, max_len=4, th_prune=9e-3, log_pr_thresh=0.01,
              type_e="Poisson", start="k-means", save_data=False,show_e=True, **kwargs):
    def go(vlhmm):
        name = path

        vlhmm.fit(data, max_len=max_len, start=start, th_prune=th_prune,
                  log_pr_thresh=log_pr_thresh, type_emission=type_e)
        print(vlhmm.tr_trie.n_contexts, vlhmm.tr_trie.seq_contexts)
        print("T=", T, "max_len=", max_len)
        comp_emission = "real emission\n{}\npredicted emission\n{} \n".format(e_params, vlhmm.emission.get_str_params())
        print(comp_emission)
        print(path)
        text = "{}\nT = {}\ninit: {}\n\n{}\n".format(type_e, T, start, comp_emission)
        for i, _vlhmm in enumerate(vlhmm.vlhmms):
            fname = "{}plot_{}".format(name, i)
            _vlhmm.plot_log_p().savefig(fname)

        fig = create_img(vlhmm, contexts, log_a, name, text)
        fig.savefig(name)

        # plt.show()

    n = len(log_a)
    h_states = ContextTransitionTrie.sample_(T, contexts, log_a)
    e_params = "unknown"
    if data is None:
        data, emission = sample_(T, n, list(map(int, h_states)), type_emission=type_e, **kwargs)
        e_params  = emission.get_str_params()
        print("real emission:\n{}".format(e_params))
        if show_e:
            emission.show()
            plt.show()
        T = len(data)
    path = "graphics/multi/{}/{}/".format(type_e, random.randrange(T))
    if not os.path.exists(path):
            os.makedirs(path)
    print(path)
    if save_data:
            data_to_file(data, path+"multi_data.txt")

    boards_parts = [0, 0.2, 0.5, 0.66, 1]
    data = get_parts(data, boards_parts)
    with open("{}info.txt".format(path), "wt") as f:
        f.write("T={}\nstart={}\nmax_len={}\nth_prune={}\nlog_pr_thresh={}\nboards_parts={}\n"
              .format(T, start, max_len, th_prune,log_pr_thresh, boards_parts))
        f.write("T {}\n\n".format([len(d) for d in data]))
        for d in data:
            f.write("{}".format(d))

    go(MultiVLHMM(n))


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

contexts = ["00", "01", "10", "110", "111"]
log_a = np.log(np.array(
    [[0.8, 0.4, 0.3, 0.2, 0.9],
     [0.2, 0.6, 0.7, 0.8, 0.1]]
))
data=None
# data = data_from_file("../tests/multi_data.txt")
main_test(contexts, log_a, T=int(4e3), max_len=3, data=data, th_prune=6e-3, start="k-means", show_e=False, save_data=True)