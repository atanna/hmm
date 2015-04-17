import os
import random
import numpy as np
import pylab as plt
from scipy.stats.mstats_basic import mquantiles
import vlhmm_.forward_backward as fb
from vlhmm_.context_tr_trie import ContextTransitionTrie
from vlhmm_.test_fb import create_img, data_to_file, sample_
from vlhmm_.multi_vlhmm import MultiVLHMM


def get_parts(data):
    T = len(data)

    t = mquantiles(list(range(T+1)), [0, 0.5, 0.66, 1])
    print(t)
    N = len(t)
    return [data[t[i-1]:t[i]] for i in range(1, N)]


def main_test(contexts, log_a, T=int(1e3), max_len=4, th_prune=9e-3, log_pr_thresh=0.01,
              type_e="Poisson", start="k-means", save_data=False,show_e=True, **kwargs):
    def go(vlhmm):
        path = "graphics/multi/{}/{}/".format(type_e, random.randrange(T))
        name = "{}{}_{}_{}_{}".format(path, start, T, th_prune, max_len)
        print(name)

        vlhmm.fit(data, max_len=max_len, start=start, th_prune=th_prune,
                  log_pr_thresh=log_pr_thresh, type_emission=type_e)
        print(vlhmm.tr_trie.n_contexts, vlhmm.tr_trie.seq_contexts)
        print("T=", T, "max_len=", max_len)
        comp_emission = "real emission\n{}\npredicted emission\n{} \n".format(e_params, vlhmm.emission.get_str_params())
        print(comp_emission)

        if not os.path.exists(path):
            os.makedirs(path)
        print(name)
        text = "{}\nT = {}\ninit: {}\n\n{}\n".format(type_e, T, start, comp_emission)
        for i, _vlhmm in enumerate(vlhmm.vlhmms):
            fname = "{}_{}".format(name, i)
            fig = create_img(_vlhmm, contexts, log_a, fname, text)
            fig.savefig(fname+".jpg")

        fig = create_img(vlhmm, contexts, log_a, name, text)
        fig.savefig(name+".jpg")
        if save_data:
            data_to_file(name+".txt")
        plt.show()

    n= len(log_a)
    h_states = ContextTransitionTrie.sample_(T, contexts, log_a)
    data, emission = sample_(T, n, list(map(int, h_states)), type_emission=type_e, **kwargs)
    data = get_parts(data)
    print(data)
    e_params  = emission.get_str_params()
    print("real emission:\n{}".format(e_params))
    if show_e:
        emission.show()
        plt.show()

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

main_test(contexts, log_a, T=1000, max_len=2, th_prune=6e-3, show_e=False)