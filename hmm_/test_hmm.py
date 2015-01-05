import numpy as np
from numpy import array
from hmm_.hmm import HMM, HMMModel


def good_test():
    data = array([0, 1, 2, 2, 1, 0, 0, 1, 2, 2, 1, 0, 0, 1, 2])
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
    data = model.sample(T)
    print("model:\n", model)
    print("sample:", data)
    print("p(data|model)",
          np.exp(HMM(n).observation_log_probability(model, data)))

    p, optimal_model = HMM(n).optimal_model(data, m=m, n_starts=5,
                                            eps=1e-17, max_iter=1e2)
    print("p(data|optimal_model)", p)
    dist = model.distance(optimal_model)
    print("dist:", np.exp(dist))
    print(optimal_model)


def fudge_test_sample():
    """
    p(data| model) = 1
    but distance between models = inf
    (because permutation of hidden sates is not invariant for model)
    """
    n, T = 3, 20
    data = array([0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1])
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


# good_test()
# fudge_test_sample()
# random_test_sample()
