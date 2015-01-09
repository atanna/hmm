import numpy as np
from hmm_.hmm import HMM, HMMModel


def good_test():
    data = np.array([0, 1, 2, 2, 1, 0, 0, 1, 2, 2, 1, 0, 0, 1, 2])
    print("data  ", data)
    for i in range(2, 8):
        p, model, h_states = HMM(i).optimal_model(data, n_starts=1,
                                                  log_eps=1e-7, max_iter=1e2)
        print("hmm({}): {}".format(i, np.exp(p)))
        if abs(p - 1.) < 1e-8:
            break
    print("h_states    ", HMM(i).optimal_state_sequence(model, data))
    print("gamma.argmax", h_states)
    print("model\n{}\n".format(model))


def test_sample(model, T, f_name=None, **kwargs):
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
    log_p = HMM.observation_log_probability(model, data)
    print("p(data|model)", log_p)

    if f_name is not None:
        with open(f_name, "w") as f:
            f.write(
                "{}\n{}\n{}\n{}\n{}\n".format(T, log_p, model, data, h_states))

    freq_model, h_states_freq = \
        HMMModel.get_freq_model_and_freq_states(data, h_states, m=m)
    print("freq_model:\n", freq_model)
    print("p(data|freq_model)",
          HMM.observation_log_probability(freq_model, data))
    print("h_states_freq:", h_states_freq)
    print("chisquare(model, freq_model):\n",
          model.chisquare(freq_model, h_states_freq))

    log_p, optimal_model, h_opt_states = HMM(n).optimal_model(data, m=m,
                                                              **kwargs)

    if f_name is not None:
        with open(f_name, 'a') as f:
            f.write("optimal:\n{}\n{}\nh_states{}"
                    .format(log_p, optimal_model, h_opt_states))

    print("optimal_model:\n", optimal_model)
    print("p(data|optimal_model)", log_p)
    print("chisquare(model, optimal_model):\n",
          model.chisquare(optimal_model, h_states_freq))
    print("dist:", model.distance(optimal_model))
    print("chisquare(freq_model, optimal_model):\n",
          freq_model.chisquare(optimal_model, h_states_freq))


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


def fudge_test_sample():
    """
    p(data| model) = 1.
    (because permutation of hidden sates is not invariant for model)
    """
    n, T = 3, 20
    data = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1])
    kwargs = {"n_starts": 5, "log_eps": 1e-17, "max_iter": 1e2}
    log_p, model, h_states = HMM(n).optimal_model(data, **kwargs)
    test_sample(model, T, **kwargs)


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
    kwargs = {"n_starts": 5, "log_eps": 1e-3, "max_iter": 1e2}
    test_sample(model, T, **kwargs)


def _test_sample():
    n, T = 2, int(1e4)
    print("T", T)
    a = np.array([[0.5, 0.5],
                  [0.3, 0.7]])
    b = np.array([[0.1, 0.9],
                  [0.3, 0.7]])
    pi = np.array([1., 0.0])
    model = HMMModel.get_model_from_real_prob(a, b, pi)
    test_sample(model, T, n_starts=1, log_eps=2 * 1e-3, max_iter=2 * 1e2)


_test_sample()
# good_fudge_test_sample()
# good_test()
# fudge_test_sample()
# random_test_sample()



