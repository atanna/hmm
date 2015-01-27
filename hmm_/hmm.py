import numpy as np
from numpy.core.umath import logaddexp
from scipy import stats
from collections import Counter
from hmm_.hmm_exception import *
from scipy.misc import logsumexp
from scipy.stats import chisquare


"""
All theory and notations were taken from:
http://www.cs.ubc.ca/~murphyk/Bayes/rabiner.pdf
http://logic.pdmi.ras.ru/~sergey/teaching/asr/notes-09-hmm.pdf
"""


class HMMModel():
    """
        let x_0 ... x_(n-1) - hidden states
            y_0 ... y_(m-1) - alphabet for observation data
            q_0 ... g_(T-1) - current hidden states
        a: shape (n,n), transition matrix for hidden states
        a[i,j] = p(q_(t+1)=x_j| q_t=x_i)
        b: shape (n,m), log_probability hidden_state -> state
        b[j,k] = p(y_k| x_j)
        pi: shape (n,), start probabilities for hidden states
        pi_j = p(q_0=x_j)
    """

    @staticmethod
    def get_model_from_real_prob(a, b, pi=None, **kwargs):
        return HMMModel.get_model(np.log(a), np.log(b),
                                  np.log(pi) if pi is not None else None,
                                  **kwargs)

    @staticmethod
    def get_model(log_a, log_b, log_pi=None, canonic=True):
        if log_pi is None:
            pi = np.random.random(len(log_a))
            log_pi = np.log(pi / pi.sum())
        model = HMMModel()
        model.log_a, model.log_b, model.log_pi = \
            np.array(log_a), np.array(log_b), np.array(log_pi)
        model.n, model.m = model.log_b.shape
        model.check()
        if canonic:
            model.canonical_form()
        return model

    @staticmethod
    def get_freq_model_and_freq_states(data, h_states, n=None, m=None):
        if n is None:
            n = len(set(h_states))
        if m is None:
            m = len(set(data))
        h_state_freq = Counter(h_states)
        b_freq = Counter(zip(h_states, data))
        b_ = np.array([[b_freq[(i, j)] / h_state_freq[i] for j in range(m)]
                       for i in range(n)])

        a_freq = Counter(zip(h_states, h_states[1:]))
        h_state_freq[h_states[-1]] -= 1
        a_ = np.array([[a_freq[(i, j)] / h_state_freq[i] for j in range(n)]
                       for i in range(n)])
        h_state_freq[h_states[-1]] += 1
        pi_ = np.zeros(n)
        pi_[data[0]] = 1.
        return HMMModel.get_model_from_real_prob(a_, b_, pi_), h_state_freq

    @staticmethod
    def get_random_model(n, m, **kwargs):
        a, b = np.random.random((n, n)), np.random.random((n, m))
        a /= a.sum(axis=1)[:, np.newaxis]
        b /= b.sum(axis=1)[:, np.newaxis]
        return HMMModel.get_model_from_real_prob(a, b, **kwargs)

    def canonical_form(self):
        order = np.lexsort(
            np.round(np.exp(self.log_b.T), 5))  # admissible error 1e-5
        self.log_a = self.log_a[order, :][:, order]
        self.log_b = self.log_b[order]
        self.log_pi = self.log_pi[order]

    def check(self, eps=1e-5):
        if self.log_a.shape != (len(self.log_pi), self.log_b.shape[0]):
            raise HMMException("wrong model's shape")
        if abs(np.exp(logsumexp(self.log_a, axis=1)) - np.ones(
                len(self.log_a))).max() > eps:
            raise HMMException("{} is not unit vector".format(
                np.exp(logsumexp(self.log_a, axis=1))))
        if abs(np.exp(logsumexp(self.log_b, axis=1)) - np.ones(
                len(self.log_a))).max() > eps:
            raise HMMException("{} is not unit vector".format(
                np.exp(logsumexp(self.log_b, axis=1))))
        if abs(np.exp(logsumexp(self.log_pi)) - np.ones(
                len(self.log_a))).max() > eps:
            raise HMMException("exp_sum(pi) must equal 1")

    def sample(self, size, with_h_states=True):
        hidden_states = np.arange(self.n)
        states = np.arange(self.m)
        data = np.zeros(size, dtype=np.int)
        h_states = np.zeros(size, dtype=np.int)
        p = self.log_pi
        for i in range(size):
            h_state = stats.rv_discrete(name='custm',
                                        values=(hidden_states,
                                                np.exp(p))).rvs()
            p = self.log_a[h_state]
            data[i] = stats.rv_discrete(name='custm',
                                        values=(states,
                                                np.exp(
                                                    self.log_b[
                                                        h_state]))).rvs()
            h_states[i] = h_state

        return data, h_states if with_h_states else data

    def chisquare(self, model, h_state_freq, alpha=0.01):
        freq = np.array([h_state_freq[i] for i
                         in range(self.n)])[:, np.newaxis]

        def arr_freq(log_arr):
            return 5 + np.exp(log_arr) * freq
            # 5 in case p=0. (in chisquare values must be >5)

        p_value_a = chisquare(arr_freq(self.log_a),
                              arr_freq(model.log_a), axis=1)[1]
        p_value_b = chisquare(arr_freq(self.log_b),
                              arr_freq(model.log_b), axis=1)[1]
        return {"a": (p_value_a > alpha).all(), "b": (p_value_b > alpha).all()}

    def distance(self, model):
        return np.mean([abs(np.exp(self.log_a) - np.exp(model.log_a)).mean(),
                        abs(np.exp(self.log_b) - np.exp(model.log_b)).mean()])

    def __str__(self):
        return "a: {}\n" \
               "b: {}\n" \
               "pi: {}".format(np.exp(self.log_a), np.exp(self.log_b),
                               np.exp(self.log_pi))


class HMM():
    def __init__(self, n):
        """
        n - number of hidden states
        """
        self.n = n

    @staticmethod
    def observation_log_probability(model, data):
        """
        :param model:
        :param data:
        :return: p(data| model)

        """
        log_p = logsumexp(HMM.forward(model, data)[-1])
        return HMM._fix_nan(log_p)

    @staticmethod
    def optimal_state_sequence(model, data):
        """
        iteration optimal algorithm
        additional values: gamma, delta, psi
        gamma[t,i] = p(q_t=x_i| data, model)
        delta[t,i] = max_(q_1...q_(t-1)) p(q_1...q_(t-1), q_t=x_i,  d_1..d_t
                          |model)
        psi[t,i] = argmax -//-
        :param model:
        :param data:
        :return: optimal q_1 ... q_T
        """

        log_delta_t = model.log_pi + model.log_b[:, data[0]]
        psi = []

        for d in data[1:]:
            tmp = log_delta_t[:, np.newaxis] + model.log_a
            psi.append(tmp.argmax(axis=0))
            log_delta_t = tmp.max(axis=0) + model.log_b[:, d]

        T = len(data)
        psi = np.array(psi)
        q = np.zeros(T)
        q[-1] = log_delta_t.argmax()
        for i in range(2, T + 1):
            q[-i] = psi[-i + 1][q[-i + 1]]
        return q.astype(np.int)

    @staticmethod
    def forward(model, data):
        """
        Forward procedure
        additional values: alpha
        alpha[t,i] = p(data[1]...data[t]| q_t=x_i, model)
        :param model:
        :param data:
        :return: alpha
        """
        T = len(data)
        log_alpha = np.zeros((T, model.n))

        log_alpha[0] = model.log_pi + model.log_b[:, data[0]]
        for t in range(1, T):
            tmp = HMM._nan_filter(logsumexp(log_alpha[t - 1] + model.log_a.T,
                                            axis=1))
            log_alpha[t] = tmp + model.log_b[:, data[t]]
        return log_alpha

    @staticmethod
    def backward(model, data):
        """
        Backward procedure
        additional values: beta
        beta[t,i] = p(data[t+1]...data[T]| q_t=x_i, model)
        :param model:
        :param data:
        :return: beta
        """
        T = len(data)
        log_beta = np.zeros((T, model.n))
        log_beta[T - 1] = np.log(1.)
        for t in range(T - 2, -1, -1):
            log_beta[t] = HMM._nan_filter(logsumexp(
                model.log_a + (model.log_b[:, data[t + 1]] + log_beta[t + 1]),
                axis=1))
        return log_beta

    @staticmethod
    def _nan_filter(arr):
        return np.array(list(map(HMM._fix_nan, arr)))

    @staticmethod
    def _fix_nan(x):
        return -np.inf if x != x else x

    def _optimal_model(self, data, start_model=None, m=None, log_eps=1e-30,
                       max_iter=1e5):
        """
        Baum-Welch algorithm
        p(data| model) ->_model max
        (it finds !!!local maximum..)
        :param data:
        :return: optimal hmm_model
        """

        def update_params(log_alpha, log_beta):
            """
            additional values: ksi, gamma
            ksi_t[i,j] = p(q_t=x_i, q_(t+1)=x_j|data, model)
            gamma[t,i] = p(q_t = x_i|data, modal)
            :param log_alpha:
            :param log_beta:
            :return:
            """
            log_a = np.log(np.zeros((n, n)))
            log_b = np.log(np.zeros((n, m)))
            log_gamma = np.log(np.zeros((T, n)))

            log_p = self._fix_nan(logsumexp(log_alpha[-1]))
            eps_check = 1e-4
            for t in range(T - 1):
                log_ksi_t = log_alpha[t][:, np.newaxis] + model.log_a \
                            + model.log_b[:, data[t + 1]] \
                            + log_beta[t + 1] - log_p
                log_gamma[t] = self._nan_filter(logsumexp(log_ksi_t, axis=1))

                log_a = logaddexp(log_a, log_ksi_t)
                log_b[:, data[t]] = logaddexp(log_b[:, data[t]], log_gamma[t])

                assert abs(np.exp(self._fix_nan(logsumexp(log_alpha[t] +
                                                             log_beta[t]))) -
                           np.exp(log_p)) < eps_check, \
                    "{} != {}".format(logsumexp(log_alpha[t] + log_beta[t]),
                                      log_p)
                assert abs(np.exp(logsumexp(log_gamma[t]))) - 1. < eps_check, \
                    "{} != 1.".format(np.exp(logsumexp(log_gamma[t])))

            log_sum_gamma = logsumexp(log_gamma, axis=0)[:, np.newaxis]
            log_a -= log_sum_gamma
            log_b -= log_sum_gamma
            log_pi = log_gamma[0]

            return HMMModel.get_model(log_a, log_b, log_pi,
                                      canonic=False), log_gamma

        n, T = self.n, len(data)
        if m is None:
            m = len(set(data))
        model = start_model
        if model is None:
            model = HMMModel.get_random_model(n, m)

        log_prev_p, log_p = 1., np.log(0.)
        n_iter = 0
        while (log_prev_p > 0 or (log_p - log_prev_p > log_eps)) \
                and n_iter < max_iter:
            log_prev_p = log_p
            log_alpha = self.forward(model, data)
            log_beta = self.backward(model, data)
            model, log_gamma = update_params(log_alpha, log_beta)
            log_p = logsumexp(log_alpha[-1])
            n_iter += 1

        return log_p, model, log_gamma

    def optimal_model(self, data, n_starts=15, start_model=None, **kwargs):

        best_log_p, best_model, best_log_gamma = \
            self._optimal_model(data, start_model, **kwargs)
        for i in range(n_starts - 1):
            log_p, model, log_gamma = self._optimal_model(data, **kwargs)
            if log_p > best_log_p:
                best_log_p, best_model, best_log_gamma = log_p, model, log_gamma
        best_model.canonical_form()
        return best_log_p, best_model, best_log_gamma.argmax(axis=1)


