import numpy as np
from functools import reduce
from scipy.cluster.vq import kmeans2
from scipy.stats import multivariate_normal


class Emission():
    def log_p(self, y, l):
        """
        :param y: observation
        :param l:
        :return: log_p(y|l)
        """
        pass


class GaussianEmission(Emission):
    def __init__(self, data, labels, n_states=None):
        if data is None:
            return
        if n_states is None:
            self.n_states = len(set(labels))
        else:
            self.n_states = n_states
        self.labels = np.array(list(labels)).astype(np.int)
        self.data = data
        self._init_params(data)

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

    def log_p(self, y, state):
        return multivariate_normal.logpdf(y, self.mu[state], self.sigma[state])

    def sample(self, state, size=1):
        return np.array(multivariate_normal.rvs(self.mu[state], self.sigma[state], size))


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




