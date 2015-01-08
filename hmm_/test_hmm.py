import numpy as np
from hmm_.hmm import HMM, HMMModel


def good_test():
    data = np.array([0, 1, 2, 2, 1, 0, 0, 1, 2, 2, 1, 0, 0, 1, 2])
    print("data  ", data)
    for i in range(2, 8):
        p, model = HMM(i).optimal_model(data, n_starts=5,
                                        eps=1e-17, max_iter=1e2)
        print("hmm({}): {}".format(i, p))
        if abs(p - 1.) < 1e-8:
            break
    print("states", HMM(i).optimal_state_sequence(model, data))
    print("model\n{}\n".format(model))


def test_sample(model, T):
    """
    sample data from model,
    count optimal_model
    compare model, optimal_model
            p(data|model) and p(data|optimal_model)
    """
    n, m = model.n, model.m
    model.canonical_form()
    data, h_states = model.sample(T)
    print("model:\n", model)
    print("p(data|model)",
          HMM.observation_log_probability(model, data))

    freq_model, h_states_freq = \
        HMMModel.get_freq_model_and_freq_states(data, h_states)
    print("freq_model:\n", freq_model)
    print("p(data|freq_model)",
          HMM.observation_log_probability(freq_model, data))
    print("chisquare(model, freq_model):\n",
          model.chisquare(freq_model, h_states_freq))

    p, optimal_model = HMM(n).optimal_model(data, m=m, n_starts=5,
                                            eps=1e-23, max_iter=1e2)

    print("optimal_model:\n", optimal_model)
    print("p(data|optimal_model)", p)
    print("chisquare(model, optimal_model):\n",
          model.chisquare(optimal_model, h_states_freq))
    print("dist:", model.distance(optimal_model))
    print("chisquare(freq_model, optimal_model):\n",
          freq_model.chisquare(optimal_model, h_states_freq))


def fudge_test_sample():
    """
    p(data| model) = 1
    but distance between models = inf
    (because permutation of hidden sates is not invariant for model)
    """
    n, T = 3, 20
    data = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1])
    p, model = HMM(n).optimal_model(data, n_starts=5,
                                    eps=1e-17, max_iter=1e2)
    test_sample(model, T)


def random_test_sample():
    """
    strange result..
    data from random model is not suitable to this model(
    """
    n, m, T = 3, 2, 20
    a, b = np.random.random((n, n)), np.random.random((n, m))
    pi = np.random.random(n)
    a /= a.sum(axis=1)[:, np.newaxis]
    b /= b.sum(axis=1)[:, np.newaxis]
    pi /= pi.sum()
    model = HMMModel.get_model_from_real_prob(a, b, pi)
    test_sample(model, T)


def good_fudge_test_sample():
    n, T = 2, 20
    a = np.array([[0., 0., 1.],
                  [1., 0., 0.],
                  [0., 1., 0.]])
    b = np.array([[1, 0., 0., 0.],
                  [0., 1., 0., 0.],
                  [0., 0., 0., 1.]])
    pi = np.array([1., 0., 0.])
    model = HMMModel.get_model_from_real_prob(a, b, pi)
    test_sample(model, T)


def _test_sample():
    n, T = 2, int(10e2)
    # T = 10
    a = np.array([[0.5, 0.5],
                  [0.3, 0.7]])
    b = np.array([[0.1, 0.9],
                  [0.3, 0.7]])
    pi = np.array([1., 0.0])
    model = HMMModel.get_model_from_real_prob(a, b, pi)
    test_sample(model, T)


_test_sample()
# good_fudge_test_sample()
# good_test()
# fudge_test_sample()
# random_test_sample()



