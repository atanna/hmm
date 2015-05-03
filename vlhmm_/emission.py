import random
import numpy as np
import pylab as plt
import scipy
from scipy.stats import multivariate_normal, poisson


class Emission():
    def __init__(self, data=None, labels=None, n_states=None):
        self.n_states = n_states
        if data is None:
            return self
        if labels is None:
            labels = [random.choice(range(n_states)) for i in range(len(data))]
        if n_states is None:
            self.n_states = len(set(labels))
        self.labels = np.array(list(labels)).astype(np.int)
        self.data = data
        self._init_params(data)

    def _init_params(self, data):
        pass

    def get_str_params(self):
        pass

    def set_rand_params(self):
        pass

    def set_canonic_view(self):
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
    def __init__(self, *args, **kwargs):
        self.name = "Gauss"
        super().__init__(*args, **kwargs)

    def _set_params(self, mu, sigma):
        self.n_states = len(mu)
        self.n = mu.shape[1]
        self.mu = mu
        self.sigma = sigma

    def _init_params(self, data, eps=1e-4):
        self.n = data.shape[1]
        self.mu = np.zeros((self.n_states, self.n))
        self.sigma = np.zeros((self.n_states, self.n, self.n))
        for state in range(self.n_states):
            y = data[self.labels == state]
            self.mu[state] = y.mean(axis=0)
            tmp = (y - self.mu[state]) + eps
            self.sigma[state] = np.array((tmp.T.dot(tmp)) / len(y)) \
                .reshape((self.n, self.n))
        self.set_canonic_view()

    def update_params(self, log_gamma, log_=True):
        def get_new_params(state):
            mu = (gamma[:, state][:, np.newaxis] * self .data).sum(axis=0)
            denom = gamma[:, state].sum()
            mu /= denom
            sigma = np.zeros((self.n, self.n))
            for t in range(T):
                tmp = (self.data[t] - mu)[:, np.newaxis]
                sigma += gamma[t, state] \
                         * tmp.dot(tmp.T)
            return mu, sigma / denom

        if log_:
            gamma = np.exp(log_gamma)
        else:
            gamma = log_gamma
        T = len(gamma)
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

    def set_canonic_view(self):
        order = self.get_order()
        self.mu = self.mu[order]
        self.sigma = self.sigma[order]

    def get_str_params(self, t_order="sorted"):
        order = list(range(self.n_states))
        if t_order == "sorted":
            order = self.get_order()
        return "mu:\n{}\nsigma:\n{}".format(np.round(self.mu[order],1),
                                            np.round(self.sigma[order],1))

    def log_p(self, y, state):
        return multivariate_normal.logpdf(y, self.mu[state], self.sigma[state])

    def all_log_p(self, y):
        return np.array([self.log_p(y, i) for i in range(self.n_states)])

    def sample(self, state, size=1):
        return np.array(multivariate_normal.rvs(self.mu[state], self.sigma[state], size))

    def get_order(self):
        return list(map(lambda x: x[1], sorted(zip(self.mu, range(self.n_states)), key=lambda x: x[0][0])))

    def show(self, rihgt_order=True, n=1000, col=0):
        if rihgt_order:
            order = self.get_order()
            mu, sigma = self.mu[order], self.sigma[order]
        else:
            mu, sigma = self.mu, self.sigma
        if mu.ndim > 1:
            mu = mu[:,col], sigma = sigma[col, col]
        fig = plt.figure()
        min_x = np.min(mu-3*sigma)
        max_x = np.max(mu+3*sigma)
        x = np.linspace(min_x, max_x, n)
        for i, (m, var) in enumerate(zip(mu, sigma)):
            y = scipy.stats.norm.pdf(x, m, var)
            plt.plot(x, y, label="state {},  ({}, {})".format(i, np.round(m,2), np.round(var,2)))
        plt.legend(loc='upper right')
        return fig


class PoissonEmission(Emission):
    """
    1-dim
    """
    def __init__(self, *args, **kwargs):
        self.name = "Poisson"
        super().__init__(*args, **kwargs)

    def get_n_params(self):
        return len(self.alpha)

    def _set_params(self, alpha=None):
        if alpha is None:
            self.set_rand_params()
            return
        self.n_states = len(alpha)
        self.alpha = alpha

    def _init_params(self, data):
        self.alpha = np.zeros(self.n_states)
        for state in range(self.n_states):
            y = data[self.labels == state]
            self.alpha[state] = y.mean()
        self.set_canonic_view()

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

    def set_rand_params(self, _var=15.):
        self.alpha = np.abs(np.random.random(self.n_states)) * _var
        self.alpha[1:] += 2*self.alpha[:-1]
        self.alpha = np.sort(self.alpha)
        return self

    def set_canonic_view(self):
        order = self.get_order()
        self.alpha = self.alpha[order]

    def get_str_params(self, t_order='sorted'):
        order = list(range(self.n_states))
        if t_order == "sorted":
            order = self.get_order()
        return "$\\lambda$ = {}".format(np.round(self.alpha[order], 1))

    def log_p(self, y, state):
        return poisson.logpmf(y, self.alpha[state])

    def all_log_p(self, y):
        return np.array([self.log_p(y, i) for i in range(self.n_states)])

    def sample(self, state, size=1):
        return np.array(poisson.rvs(self.alpha[state], size=size))

    def get_order(self):
        return list(map(lambda x: x[1],
                        sorted(zip(self.alpha, range(self.n_states)),
                               key=lambda x: x[0])))

    def show(self, right_order=True):
        if right_order:
            alpha = self.alpha[self.get_order()]
        else:
            alpha = self.alpha
        fig = plt.figure()
        ax = fig.add_subplot("111")
        max_x = int(2*max(alpha))+1
        x = np.array(list(range(max_x)))
        for i, alph in enumerate(alpha):
            y = scipy.stats.poisson.pmf(x, alph)
            ax.plot(x, y, "o",  label="state {}, $\\lambda$ = {}"
                    .format(i, np.round(alph, 2)))
            ax.set_title("Emission density functions")
            ax.set_xlabel("x")
            ax.set_ylabel('p(x)')
            ax.vlines(x, [0], y)
        ax.legend(loc='upper right')
        return fig






