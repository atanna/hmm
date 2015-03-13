import sys
import numpy as np
from hmm_.hmm import HMMModel, HMM
import vlhmm_.forward_backward as fb
from vlhmm_.test import get_data
from vlhmm_.vlhmm_wang import VLHMMWang


def data_to_file(data, f_name):
    np.savetxt(f_name, data)


def data_from_file(f_name):
    return np.genfromtxt(f_name)

def test_wang():
    sys.stdout = open("test/wang", 'w')
    data = get_data(100)
    # show_data(data)
    n = 2
    vlhmm = VLHMMWang(n)
    X = vlhmm.fit(data, max_len=3, n_iter=3)
    print(vlhmm.context_trie.contexts)

    print("\n\n")

    vlhmm = fb.VLHMMWang(n)
    vlhmm.fit(data, X=X, max_len=3, n_iter=3)
    print(vlhmm.context_trie.contexts)


test_wang()


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
