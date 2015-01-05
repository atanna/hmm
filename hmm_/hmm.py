import numpy as np
from scipy import stats
from hmm_.hmm_exception import *


"""
All theory and notations were taken from:
http://www.cs.ubc.ca/~murphyk/Bayes/rabiner.pdf
http://logic.pdmi.ras.ru/~sergey/teaching/asr/notes-09-hmm.pdf
"""


class HMMModel():
    def __init__(self, a, b, pi):
        """
        let x_0 ... x_(n-1) - hidden states
            y_0 ... y_(m-1) - alphabet for observation data
            q_0 ... g_(T-1) - current hidden states
        :param a:
        shape = (n,n)
        a[i,j] = p(q_(t+1)=x_j| q_t=x_i)
        :param b:
        shape = (n,m)
        b[j,k] = p(y_k| x_j)
        :param pi:
        shape = (n,)
        pi_j = p(q_0=x_j)
        :return:
        """

        self.a = np.array(a)
        self.b = np.array(b)
        self.pi = np.array(pi)
        self.n, self.m = self.b.shape
        self.check()

    def check(self, eps=1e-10):
        if self.a.shape != (len(self.pi), self.b.shape[0]):
            raise HMMException("wrong model's shape")
        if abs(self.a.sum(axis=1) - np.ones(len(self.a))).max() > eps:
            raise HMMException("a.sum(axis=1) is not unit vector")
        if abs(self.b.sum(axis=1) - np.ones(len(self.a))).max() > eps:
            raise HMMException("b.sum(axis=1) is not unit vector")
        if abs(sum(self.pi) - np.ones(len(self.a))).max() > eps:
            raise HMMException("sum(pi) must equal 1")

    def sample(self, size):
        hidden_states = np.arange(self.n)
        states = np.arange(self.m)
        sample = np.zeros(size, dtype=np.int)
        p = self.pi
        for i in range(size):
            h_state = stats.rv_discrete(name='custm',
                                        values=(hidden_states, p)).rvs()
            sample[i] = stats.rv_discrete(name='custm',
                                          values=(
                                          states, self.b[h_state])).rvs()
            p = self.a[h_state]
        return sample

    def distance(self, model):
        return np.mean([abs(self.a - model.a).mean(),
                        abs(self.b - model.b).mean()])

    def __str__(self):
        return "a: {}\n" \
               "b: {}\n" \
               "pi: {}\n".format(self.a, self.b, self.pi)


class HMM():
    def __init__(self, n):
        """
        n - number of hidden states
        """
        self.n = n

    @staticmethod
    def observation_probability(model, data):
        """
        :param model:
        :param data:
        :return: p(data| model)

        """
        return HMM.forward(model, data)[-1].sum()

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

        delta_t = model.pi * model.b[:, data[0]]
        psi = []

        for d in data[1:]:
            tmp = delta_t[:, np.newaxis] * model.a
            psi.append(tmp.argmax(axis=0))
            delta_t = tmp.max(axis=0) * model.b[:, d]

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

        alpha[0] = model.pi * model.b[:, data[0]]
        for t in range(1, T):
            alpha[t] = alpha[t - 1].dot(model.a) * (model.b[:, data[t]])
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
        beta[T - 1] = 1.
        for t in range(T - 2, -1, -1):
            beta[t] = model.a.dot(model.b[:, data[t + 1]] * beta[t + 1])
        return beta

    def _optimal_model(self, data, start_model=None, eps=1e-30, max_iter=1e5):
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
            gamma = p(q_t = x_i| D, alpha)
            :param alpha:
            :param beta:
            :return:
            """
            a = np.zeros((n, n))
            b = np.zeros((n, m))
            gamma = np.zeros((T, n))

            p = alpha[-1].sum()
            for t in range(T - 1):
                ksi_t = alpha[t][:, np.newaxis] * model.a \
                        * model.b[:, data[t + 1]] * beta[t + 1] / p
                gamma[t] = ksi_t.sum(axis=1)

                a += ksi_t
                b[:, data[t]] += gamma[t]

            sum_gamma = gamma.sum(axis=0)[:, np.newaxis]
            a /= sum_gamma
            b /= sum_gamma
            pi = gamma[0]

            return HMMModel(a, b, pi), gamma

        n, m, T = self.n, len(set(data)), len(data)
        model = start_model
        if model is None:
            a, b, pi = np.random.random((n, n)), np.random.random((n, m)), \
                       np.random.random(n)
            a /= a.sum(axis=1)[:, np.newaxis]
            b /= b.sum(axis=1)[:, np.newaxis]
            pi /= pi.sum()
            model = HMMModel(a, b, pi)

        prev_p, p = -1., 0.
        n_iter = 0
        while abs(p - prev_p) > eps and n_iter < max_iter:
            alpha, beta = self.forward(model, data), self.backward(model, data)
            model, gamma = update_params(alpha, beta)
            p = alpha[-1].sum()
            n_iter += 1

        return p, model

    def optimal_model(self, data, n_starts=15, start_model=None, **kwargs):

        best_p, best_model = self._optimal_model(data, start_model, **kwargs)
        for i in range(n_starts):
            p, model = self._optimal_model(data, **kwargs)
            if p > best_p:
                best_p, best_model = p, model
        return best_p, best_model


