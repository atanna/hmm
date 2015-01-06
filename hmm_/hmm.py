import numpy as np
from scipy import stats
from hmm_.hmm_exception import *
from scipy.misc import logsumexp


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
        a[i,j] = log_p(q_(t+1)=x_j| q_t=x_i)
        b: shape (n,m), log_probability hidden_state -> state
        b[j,k] = log_p(y_k| x_j)
        pi: shape (n,), start log_probabilities for hidden states
        pi_j = log_p(q_0=x_j)
    """

    @staticmethod
    def get_model_from_real_prob(a, b, pi, **kwargs):
        return HMMModel.get_model(np.log(a), np.log(b), np.log(pi), **kwargs)

    @staticmethod
    def get_model(a, b, pi, canonic=True):
        model = HMMModel()
        model.a, model.b, model.pi = np.array(a), np.array(b), np.array(pi)
        model.n, model.m = model.b.shape
        model.check()
        if canonic:
            model.canonical_form()
        return model

    def canonical_form(self):
        order = np.lexsort(
            np.round(np.exp(self.b.T), 5))  # admissible error 1e-5
        self.a = self.a[order, :][:, order]
        self.b = self.b[order]
        self.pi = self.pi[order]

    def check(self, eps=1e-10):
        if self.a.shape != (len(self.pi), self.b.shape[0]):
            raise HMMException("wrong model's shape")
        if abs(np.exp(logsumexp(self.a, axis=1)) - np.ones(
                len(self.a))).max() > eps:
            raise HMMException("exp_a.sum(axis=1) is not unit vector")
        if abs(np.exp(logsumexp(self.b, axis=1)) - np.ones(
                len(self.a))).max() > eps:
            raise HMMException("exp_b.sum(axis=1) is not unit vector")
        if abs(np.exp(logsumexp(self.pi)) - np.ones(len(self.a))).max() > eps:
            raise HMMException("exp_sum(pi) must equal 1")

    def sample(self, size):
        hidden_states = np.arange(self.n)
        states = np.arange(self.m)
        sample = np.zeros(size, dtype=np.int)
        h_sample = np.zeros(size, dtype=np.int)
        p = self.pi
        for i in range(size):
            h_state = stats.rv_discrete(name='custm',
                                        values=(hidden_states,
                                                np.exp(p))).rvs()
            p = self.a[h_state]
            sample[i] = stats.rv_discrete(name='custm',
                                          values=(states,
                                                  np.exp(
                                                      self.b[h_state]))).rvs()
            h_sample[i] = h_state

        return sample, h_sample

    def distance(self, model):
        return np.mean([abs(np.exp(self.a) - np.exp(model.a)).mean(),
                        abs(np.exp(self.b) - np.exp(model.b)).mean()])

    def __str__(self):
        return "a: {}\n" \
               "b: {}\n" \
               "pi: {}\n".format(np.exp(self.a), np.exp(self.b),
                                 np.exp(self.pi))


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
        p = logsumexp(HMM.forward(model, data)[-1])
        return -np.inf if p != p else p  # fix nan

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

        delta_t = model.pi + model.b[:, data[0]]
        psi = []

        for d in data[1:]:
            tmp = delta_t[:, np.newaxis] + model.a
            psi.append(tmp.argmax(axis=0))
            delta_t = tmp.max(axis=0) + model.b[:, d]

        T = len(data)
        psi = np.array(psi)
        q = np.zeros(T)
        q[T - 1] = delta_t.argmax()
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
        alpha = np.zeros((T, model.n))

        alpha[0] = model.pi + model.b[:, data[0]]
        for t in range(1, T):
            tmp = logsumexp(alpha[t - 1] + model.a.T, axis=1)
            tmp = np.array(list(map(lambda t: -np.inf if t != t else t, tmp)))
            alpha[t] = tmp + model.b[:, data[t]]
        return alpha

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
        beta = np.zeros((T, model.n))
        beta[T - 1] = np.log(1.)
        for t in range(T - 2, -1, -1):
            beta[t] = logsumexp(
                model.a + (model.b[:, data[t + 1]] + beta[t + 1]), axis=1)
        return beta

    def _optimal_model(self, data, start_model=None, m=None, eps=1e-30,
                       max_iter=1e5):
        """
        Baum-Welch algorithm
        p(data| model) ->_model max
        (it finds !!!local maximum..)
        :param data:
        :return: optimal hmm_model
        """

        def update_params(alpha, beta):
            """
            additional values: ksi, gamma
            ksi_t[i,j] = p(q_t=x_i, q_(t+1)=x_j|data, model)
            gamma[t,i] = p(q_t = x_i|data, modal)
            :param alpha:
            :param beta:
            :return:
            """
            a = np.log(np.zeros((n, n)))
            b = np.log(np.zeros((n, m)))
            gamma = np.log(np.zeros((T, n)))

            p = logsumexp(alpha[-1])
            eps_check = 1e-8
            for t in range(T - 1):
                ksi_t = alpha[t][:, np.newaxis] + model.a \
                        + model.b[:, data[t + 1]] + beta[t + 1] - p
                gamma[t] = logsumexp(ksi_t, axis=1)

                a = logsumexp([a, ksi_t], axis=0)
                b[:, data[t]] = logsumexp([b[:, data[t]], gamma[t]], axis=0)

                assert abs(np.exp(logsumexp(alpha[t] + beta[t])) -
                           np.exp(p)) < eps_check, \
                    "{} != {}".format(logsumexp(alpha[t] + beta[t]), p)
                assert abs(np.exp(logsumexp(gamma[t])) - 1.) < eps_check, \
                    "{} != 1.".format(np.exp(logsumexp(gamma[t])))

            sum_gamma = logsumexp(gamma, axis=0)[:, np.newaxis]
            a -= sum_gamma
            b -= sum_gamma
            pi = gamma[0]

            return HMMModel.get_model(a, b, pi, canonic=False), gamma

        n, T = self.n, len(data)
        if m is None:
            m = len(set(data))
        model = start_model
        if model is None:
            a, b, pi = np.random.random((n, n)), np.random.random((n, m)), \
                       np.random.random(n)
            a /= a.sum(axis=1)[:, np.newaxis]
            b /= b.sum(axis=1)[:, np.newaxis]
            pi /= pi.sum()
            model = HMMModel.get_model_from_real_prob(a, b, pi, canonic=False)

        prev_p, p = np.log(1.), np.log(0.)
        n_iter = 0
        while abs(np.exp(p) - np.exp(prev_p)) > eps and n_iter < max_iter:
            alpha, beta = self.forward(model, data), self.backward(model, data)
            model, gamma = update_params(alpha, beta)
            p = logsumexp(alpha[-1])
            n_iter += 1

        return np.exp(p), model

    def optimal_model(self, data, n_starts=15, start_model=None, **kwargs):

        best_p, best_model = self._optimal_model(data, start_model, **kwargs)
        for i in range(n_starts):
            p, model = self._optimal_model(data, **kwargs)
            if p > best_p:
                best_p, best_model = p, model
        best_model.canonical_form()
        return best_p, best_model


