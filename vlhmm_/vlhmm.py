import random
import numpy as np
from functools import reduce
from scipy.cluster.vq import kmeans2
from scipy.stats import multivariate_normal, poisson


class Emission():
    def __init__(self, data=None, labels=None, n_states=None):
        self.n_states = n_states
        if data is None:
            return
        if labels is None:
            labels = [random.choice(range(n_states)) for i in range(len(data))]
        if n_states is None:
            self.n_states = len(set(labels))
        self.labels = np.array(list(labels)).astype(np.int)
        self.data = data
        self._init_params(data)

    def _init_params(self, data):
        pass

    def set_rand_params(self):
        pass

    def get_order(self):
        pass

    def log_p(self, y, l):
        """
        :param y: observation
        :param l:
        :return: log_p(y|l)
        """
        pass


class GaussianEmission(Emission):
    def _set_params(self, mu, sigma):
        self.n_states = len(mu)
        self.n = mu.shape[1]
        self.mu = mu
        self.sigma = sigma

    def _init_params(self, data, eps=1e-4):
        self.n = data.shape[1]
        self.mu = np.zeros((self.n_states, self.n))
        self.sigma = np.zeros((self.n_states, self.n, self.n))
        self.T = len(data)
        for state in range(self.n_states):
            y = data[self.labels == state]
            self.mu[state] = y.mean(axis=0)
            tmp = (y - self.mu[state]) + eps
            self.sigma[state] = np.array((tmp.T.dot(tmp)) / len(y)) \
                .reshape((self.n, self.n))

    def update_params(self, log_gamma, log_=True):
        def get_new_params(state):
            mu = (gamma[:, state][:, np.newaxis] * self .data).sum(axis=0)
            denom = gamma[:, state].sum()
            mu /= denom
            sigma = np.zeros((self.n, self.n))
            for t in range(self.T):
                tmp = (self.data[t] - mu)[:, np.newaxis]
                sigma += gamma[t, state] \
                         * tmp.dot(tmp.T)
            return mu, sigma / denom

        if log_:
            gamma = np.exp(log_gamma)
        else:
            gamma = log_gamma
        for state in range(self.n_states):
            self.mu[state], self.sigma[state] = get_new_params(state)

    def set_rand_params(self, n=2, _var=10.):
        self.n = n
        mu = []
        sigma = []
        for i in range(self.n_states):
            mu.append(([random.randrange(_var / self.n),
                     random.randrange(_var / self.n)] + np.random.random((self.n,))) * _var)
            sigma.append(np.random.random((self.n, self.n)) * _var)
        self.mu  = np.array(sorted(mu, key=lambda x: x[0]))
        self.sigma = np.array(sigma)

    def log_p(self, y, state):
        return multivariate_normal.logpdf(y, self.mu[state], self.sigma[state])

    def all_log_p(self, y):
        return np.array([self.log_p(y, i) for i in range(self.n_states)])

    def sample(self, state, size=1):
        return np.array(multivariate_normal.rvs(self.mu[state], self.sigma[state], size))

    def get_order(self):
        return list(map(lambda x: x[1], sorted(zip(self.mu, range(self.n_states)), key=lambda x: x[0][0])))


class PoissonEmission(Emission):
    """
    1-dim
    """

    def _set_params(self, alpha):
        self.n_states = len(alpha)
        self.alpha = alpha

    def _init_params(self, data):
        self.alpha = np.zeros(self.n_states)
        self.T = len(data)
        for state in range(self.n_states):
            y = data[self.labels == state]
            self.alpha[state] = y.mean()

    def update_params(self, log_gamma, log_=True):
        def get_new_params(state):
            alpha = (gamma[:, state][:,np.newaxis] * self.data).sum()
            denom = gamma[:, state].sum()
            return alpha / denom

        if log_:
            gamma = np.exp(log_gamma)
        else:
            gamma = log_gamma
        for state in range(self.n_states):
            self.alpha[state] = get_new_params(state)

    def set_rand_params(self, _var=10.):
        self.alpha = np.abs(np.random.random(self.n_states)) * _var
        self.alpha = np.sort(self.alpha)

    def log_p(self, y, state):
        return poisson.logpmf(y, self.alpha[state])

    def all_log_p(self, y):
        return np.array([self.log_p(y, i) for i in range(self.n_states)])

    def sample(self, state, size=1):
        return np.array(poisson.rvs(self.alpha[state], size=size))

    def get_order(self):
        return list(map(lambda x: x[1], sorted(zip(self.alpha, range(self.n_states)), key=lambda x: x[0])))

class AbstractVLHMM():
    def __init__(self, n):
        """
        :param n: - length of context's alphabet, n <= 10
        :return:
        """
        self.n = n

    def _init_X(self, data):
        try:
            centre, labels = kmeans2(data, self.n)
        except TypeError:
            labels= np.random.choice(range(self.n), len(data))
        self.X = "".join(list(map(str, labels)))

    def get_c(self, *args):
        return self.tr_trie.get_c(
            reduce(lambda res, x: res + str(x), args, ""))




