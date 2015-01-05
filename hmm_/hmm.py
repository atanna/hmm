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
                                          values=(states, self.b[h_state])).rvs()
            p = self.a[h_state]
        return sample

    def distance(self, model):
        return np.mean([abs(self.a - model.a).mean(),
                       abs(self.b - model.b).mean(),
                       abs(self.pi - model.pi).mean()])

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

    def observation_probability(self, model, data):
        """
        Forward-backward procedure
        additional values: alpha
        alpha[t,i] = p(data[1]...data[t]| q_t=x_i, model)
        :param model:
        :param data:
        :return: p(data| model)

        """
        alpha_t = model.pi * model.b[:, data[0]]

        for d in data[1:]:
            alpha_t = alpha_t.dot(model.a) * (model.b[:, d])

        return alpha_t.sum()

    def optimal_state_sequence(self, model, data):
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
        q[T-1] = delta_t.argmax()
        for i in range(2, T+1):
            q[-i] = psi[-i+1][q[-i + 1]]
        return q.astype(np.int)

    def _optimal_model(self, data, start_model=None, eps=1e-30, max_iter=1e5):
        """
        Baum-Welch algorithm
        p(data| model) ->_model max
        (it finds !!!local maximum..)
        additional values: ksi, gamma
        ksi_t[i,j] = p(q_t=x_i, q_(t+1)=x_j|data, model)
        :param data:
        :return: optimal hmm_model
        """

        def em(p, eps_check = 1e-5):
            nonlocal model
            beta = np.zeros((T, n))
            a = np.zeros((n, n))
            b = np.zeros((n, m))
            pi = np.zeros(n)
            gamma_t = np.zeros(n)
            ksi_t = None

            beta[T - 1] = 1.
            for t in range(T - 2, -1, -1):
                beta[t] = model.a.dot(model.b[:, data[t + 1]] * beta[t + 1])


            def do(t):
                nonlocal a, b, ksi_t, gamma_t, alpha_t
                ksi_t = alpha_t[:, np.newaxis] * model.a \
                    * model.b[:, data[t+1]] * beta[t+1] / p

                gamma_t = ksi_t.sum(axis=1)
                b[:, data[t]] += gamma_t
                a += ksi_t
                assert abs(alpha_t.dot(beta[t]) - p) < eps_check, \
                    "{} != p".format(alpha_t.dot(beta[t]))
                assert abs(gamma_t.sum() - 1.) < eps_check, \
                    "{} != 1.".format(gamma_t.sum())
                assert abs((alpha_t * beta[t] / p).sum() - 1.) < eps_check, \
                    "{} != 1.".format((alpha_t * beta[t] / p).sum())

            alpha_t = model.pi * model.b[:, data[0]]
            do(0)
            pi = gamma_t.copy()
            sum_gamma = gamma_t.copy()


            for t in range(1, T-1):
                alpha_t = alpha_t.dot(model.a) * (model.b[:, data[t]])
                do(t)
                sum_gamma += gamma_t

            alpha_t = alpha_t.dot(model.a) * (model.b[:, data[-1]])
            assert abs(alpha_t.sum() - p) < eps_check, \
                "alpha_t.sum() = {}  !=  p = {}".format(alpha_t.sum(), p)

            sum_gamma = sum_gamma[:, np.newaxis]
            model = HMMModel(a / sum_gamma, b / sum_gamma, pi)
            p = self.observation_probability(model, data)
            return p

        n, m, T = self.n, len(set(data)), len(data)
        model = start_model
        if model is None:
            a, b, pi = np.random.random((n, n)), np.random.random((n, m)), \
                       np.random.random(n)
            a /= a.sum(axis=1)[:, np.newaxis]
            b /= b.sum(axis=1)[:, np.newaxis]
            pi /= pi.sum()
            model = HMMModel(a, b, pi)

        p_prev, p_curr = 0., self.observation_probability(model, data)

        n_iter = 0
        while abs(p_curr - p_prev) > eps and n_iter < max_iter:
            p_curr, p_prev = em(p_curr), p_curr
            n_iter += 1

        return p_curr, model

    def optimal_model(self, data, n_starts=15, start_model=None, **kwargs):

        best_p, best_model = self._optimal_model(data, start_model, **kwargs)
        for i in range(n_starts):
            p, model = self._optimal_model(data, **kwargs)
            if p > best_p:
                best_p, best_model = p, model
        return best_p, best_model


