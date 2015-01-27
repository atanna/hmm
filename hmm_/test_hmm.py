import sys
from sklearn.hmm import MultinomialHMM
from hmm_.hmm import HMM, HMMModel


def sklern_test(data, **kwargs):
    hmm = MultinomialHMM(**kwargs)
    hmm.fit([data])
    return HMMModel.get_model(hmm._log_transmat, hmm._log_emissionprob,
                              hmm._log_startprob), hmm.score(data), hmm


def test_sample(model, T, f_name=None, **kwargs):
    """
    sample data from model
    count frequency_model, sklearn_model, optimal_model
    compare them
    """

    n, m = model.n, model.m
    model.canonical_form()

    print("T", T)
    data, h_states = model.sample(T)
    model, log_sklearn_p, hmm = sklern_test(data, n_components=n,
                                               n_iter=int(kwargs['max_iter']))
    data, h_states = hmm.sample(T)
    print(data)

    print("model:\n", model)
    log_real_p = HMM.observation_log_probability(model, data)
    print("p(data|model)", log_real_p)

    if f_name is not None:
        with open(f_name, "w") as f:
            f.write(
                "{}\n{}\n{}\n{}\n{}\n".format(T, log_real_p, model,
                                              data, h_states))

    freq_model, h_states_freq = \
        HMMModel.get_freq_model_and_freq_states(data, h_states, m=m)
    log_freq_p = HMM.observation_log_probability(freq_model, data)
    print("\nfreq_model:\n", freq_model)
    print("p(data|freq_model)", log_freq_p)
    print("h_states_freq:", h_states_freq)
    print("chisquare(model, freq_model):",
          model.chisquare(freq_model, h_states_freq))

    sklearn_model, log_sklearn_p, hmm = sklern_test(data, n_components=n,
                                               n_iter=int(kwargs['max_iter']))

    print("\nsklearn_model:\n", sklearn_model)
    print("p(data|sklearn_model)", log_sklearn_p)
    print("dist:", model.distance(sklearn_model))
    print("chisquare(model, sklearn_model):",
          model.chisquare(sklearn_model, h_states_freq))
    eps_check = 1e-5
    assert (abs(HMM.observation_log_probability(sklearn_model, data)
                - log_sklearn_p) < eps_check)

    log_p, optimal_model, h_opt_states = HMM(n).optimal_model(data, m=m,
                                                              **kwargs)

    if f_name is not None:
        with open(f_name, 'a') as f:
            f.write("optimal:\n{}\n{}\nh_states{}"
                    .format(log_p, optimal_model, h_opt_states))

    print("\noptimal_model:\n", optimal_model)
    print("p(data|optimal_model)", log_p)
    print("chisquare(model, optimal_model):",
          model.chisquare(optimal_model, h_states_freq))
    print("dist:", model.distance(optimal_model))

    print("\n")
    print("chisquare(sklearn_model, optimal_model):",
          sklearn_model.chisquare(optimal_model, h_states_freq))
    print("dist:", sklearn_model.distance(optimal_model))

    print("p, freq_p, sklearn_p, optimal_p:",
          log_real_p, log_freq_p, log_sklearn_p, log_p)


def _test_sample():
    n, m, T = 2, 2, int(1e3)

    model = HMMModel.get_random_model(n, m)

    sys.stdout = open("test/{}_{}_{}_test_sklearn_sample".format(n, m, T), 'w')
    test_sample(model, T, n_starts=3, log_eps=2e-3, max_iter=1e2)


_test_sample()



