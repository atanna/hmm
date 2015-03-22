from scipy import stats
import numpy as np
from scipy.misc import logsumexp
from hmm_.hmm import HMMModel
from vlhmm_.context_tr_trie import ContextTransitionTrie
from vlhmm_.vlhmm import AbstractVLHMM, GaussianEmission


class AbstractForwardBackward():
    def _init(self, data):
        self.T = len(data)
        self.data = data
        if data.ndim < 2:
            self.data = data[:, np.newaxis]
        self._init_a()
        self._init_emission()
        self.n_contexts = self._get_n_contexts()
        self.log_alpha = np.zeros((self.T, self.n_contexts))
        self.log_beta = np.zeros((self.T, self.n_contexts))
        self.log_gamma = np.zeros((self.T, self.n_contexts))
        self.log_ksi = np.zeros((self.T, self.n, self.n_contexts))

    def _init_X(self, data):
        pass

    def _get_n_contexts(self):
        pass

    def _init_a(self):
        pass

    def _init_emission(self):
        pass

    def fit(self, data, n_iter, X=None):
        if X is not None:
            self.X = X
        else:
            self._init_X(data)
        self._init(data)
        self._em(n_iter)
        return self

    def _em(self, n_iter):
        def e_step():
            self.log_forward()
            self.log_backward()
            self._log_gamma()

        def m_step():
            self.update_tr_params()
            self.update_emission_params()

        for i in range(n_iter):
            e_step()
            m_step()

    def _update_first_alpha(self, i):
        pass

    def _update_next_alpha(self, t, i):
        pass

    def _update_beta(self, t, i):
        pass

    def _update_ksi(self, t, q, i):
        pass

    def log_forward(self):
        self.log_alpha[:] = -np.inf

        for i in range(self.n_contexts):
            self._update_first_alpha(i)
        for t in range(self.T - 1):
            for i in range(self.n_contexts):
                self._update_next_alpha(t, i)

    def log_backward(self):
        self.log_beta[:] = np.log(0.)
        self.log_beta[-1] = np.log(1.)
        for t in range(self.T - 2, -1, -1):
            for i in range(self.n_contexts):
                self._update_beta(t, i)

    def _log_gamma(self):
        self._log_p = logsumexp(self.log_alpha[-1])
        self.log_gamma = self.log_alpha + self.log_beta
        self.log_gamma -= logsumexp(self.log_gamma, axis=1)[:, np.newaxis]

    def _normalise_a(self):
        self.log_a = self.log_a - logsumexp(self.log_a, axis=0)

    def update_emission_params(self):
        self.emission.update_params(self.log_gamma)

    def update_tr_params(self):
        self.log_ksi[:] = np.log(0.)
        for t in range(self.T - 1):
            for i in range(self.n_contexts):
                for q in range(self.n):
                    self._update_ksi(t, q, i)

        log_a = np.zeros((self.n, self.n_contexts))
        log_a[:] = np.log(0.)
        for i in range(self.n_contexts):
            sum_gamma = logsumexp(self.log_gamma[:, i])
            for q in range(self.n):
                log_a[q, i] = logsumexp(self.log_ksi[:, q, i]) - sum_gamma

        log_a = log_a - logsumexp(log_a, axis=0)
        self.log_a = log_a
        print("a:\n{}".format(np.exp(self.log_a)))
        print("log_p {}".format(self._log_p))


class VLHMMWang(AbstractVLHMM, AbstractForwardBackward):
    def _init(self, data):
        self.tr_trie = ContextTransitionTrie(self.X, max_len=self.max_len)
        self.contexts = self.tr_trie.seq_contexts
        self.state_c = list(map(lambda c: int(c[0]), self.contexts))
        self.n_contexts = self.tr_trie.n_contexts
        self.data_contexts = list(self.tr_trie.get_contexts(self.X))
        self.id_c = dict(zip(self.contexts, range(self.n_contexts)))
        self.log_context_p = np.log(np.ones(self.n_contexts) / self.n_contexts)

        super()._init(data)

        print("X: {}".format(self.X))
        print("n_contexts:", self.n_contexts)
        print("contexts: {}".format(self.contexts))
        print("data_contexts: {}".format(self.data_contexts))
        print("a: \n{}".format(np.exp(self.log_a)))
        print("init_mu = {}..\ninit_sigma = {}..\n".format(
            self.emission.mu, self.emission.sigma))

    def _init_emission(self):
        self.emission = GaussianEmission(self.data, self.X, self.n)

    def _get_n_contexts(self):
        return self.n_contexts

    def _init_a(self, uniform=False):
        self.log_a = self.tr_trie.count_log_a(uniform)

    def fit(self, data, max_len=4, X=None, n_iter=5, th_prune=1e-2):
        self.max_len = max_len
        super().fit(data, n_iter, X)
        self.tr_trie.recount_with_log_a(self.log_a, self.contexts)
        while self._prune(th_prune):
            self._em(n_iter)
            self.tr_trie.recount_with_log_a(self.log_a, self.contexts)
        return self

    def sample(self, size, start=0):
        X = np.zeros((size, self.emission.n))
        c = self.contexts[start]
        index_c = self.id_c[c]
        states = np.arange(self.n)
        for i in range(size):
            X[i] = self.emission.sample(self.state_c[index_c])
            q = stats.rv_discrete(name='custm',
                                  values=(states,
                                          np.exp(
                                              self.log_a[:, index_c]))).rvs()
            c = self.get_c(q, c)
            index_c = self.id_c[c]
        return X

    def _prune(self, th_prune):
        prune = changes = self.tr_trie.prune(th_prune)
        while changes:
            print("prune", changes)
            print(self.tr_trie.seq_contexts)
            if len(self.tr_trie.seq_contexts) == 1:
                self.log_a = self.tr_trie.count_log_a()
                return False
            self.update_contexts()
            changes = self.tr_trie.prune(th_prune)
        return prune

    def update_contexts(self):
        contexts = list(self.tr_trie.seq_contexts)
        n_contexts = len(contexts)
        id_c = dict(zip(contexts, range(n_contexts)))
        self.log_a = self.tr_trie.count_log_a()
        self.contexts = contexts
        self.state_c = list(map(lambda c: int(c[0]), self.contexts))
        self.n_contexts = n_contexts
        self.id_c = id_c
        self.log_alpha = np.zeros((self.T, self.n_contexts))
        self.log_beta = np.zeros((self.T, self.n_contexts))
        self.log_gamma = np.zeros((self.T, self.n_contexts))
        self.log_ksi = np.zeros((self.T, self.n, self.n_contexts))
        self.log_context_p = np.log(np.ones(n_contexts) / n_contexts)

    def _update_first_alpha(self, i):
        self.log_alpha[0][i] = self.log_context_p[i] \
                               + self.emission.log_p(self.data[0],
                                                     self.state_c[i])

    def _update_next_alpha(self, t, i):
        c = self.contexts[i]
        for q in range(self.n):
            c_ = self.get_c(c[1:], q)
            j = self.id_c[c_]
            self.log_alpha[t + 1][i] = np.logaddexp(
                self.log_alpha[t + 1][i],
                self.log_alpha[t][j]
                + self.log_a[int(c[0]), j]
                + self.emission.log_p(self.data[t + 1], self.state_c[i]))

    def _update_beta(self, t, i):
        c = self.contexts[i]
        for q in range(self.n):
            c_ = self.get_c(q, c)
            j = self.id_c[c_]
            self.log_beta[t][i] = np.logaddexp(
                self.log_beta[t][i],
                self.log_a[q, i]
                + self.emission.log_p(self.data[t + 1], self.state_c[j])
                + self.log_beta[t + 1][j])

    def _log_gamma(self):
        super()._log_gamma()
        self.log_context_p = logsumexp(self.log_gamma, axis=0)
        self.log_context_p -= logsumexp(self.log_context_p)

    def _update_ksi(self, t, q, i):
        i_ = self.id_c[self.get_c(q, self.contexts[i])]
        self.log_ksi[t][q, i] = self.log_alpha[t][i] + self.log_a[q, i]
        + self.emission.log_p(self.data[t + 1], self.state_c[i_]) \
        + self.log_beta[t + 1][i_] - self._log_p

    def update_emission_params(self):
        gamma = np.zeros((self.T, self.n))
        for i in range(self.n_contexts):
            q = self.state_c[i]
            gamma[:, q] += np.exp(self.log_gamma[:, i])
        gamma /= np.sum(gamma, axis=1)[:, np.newaxis]
        self.emission.update_params(gamma, log_=False)


class HMM(AbstractForwardBackward):
    def __init__(self, n):
        self.n = n

    def _init(self, data):
        self.m = len(set(data))
        self.model = HMMModel.get_random_model(self.n, self.m)
        super()._init(data)

    def _get_n_contexts(self):
        return self.n

    def _update_first_alpha(self, i):
        self.log_alpha[0][i] = self.model.log_pi[i] + self.model.log_b[
            i, self.data[0]]

    def _update_next_alpha(self, t, j):
        tmp = logsumexp(self.log_alpha[t] + self.model.log_a[:, j])
        self.log_alpha[t + 1][j] = tmp + self.model.log_b[j, self.data[t + 1]]

    def _update_beta(self, t, i):
        self.log_beta[t][i] = logsumexp(
            self.model.log_a[i] + (
            self.model.log_b[:, self.data[t + 1]] + self.log_beta[t + 1]))

    def _update_ksi(self, t, q, i):
        self.log_ksi[t][q, i] = self.log_alpha[t][i] + self.model.log_a \
                                + self.model.log_b[:, self.data[t + 1]] \
                                + self.log_beta[t + 1] - self._log_p

    def update_emission_params(self):
        log_b = np.log(np.zeros((self.n, self.m)))
        for t in range(self.T - 1):
            log_b[:, self.data[t]] = np.logaddexp(log_b[:, self.data[t]],
                                                  self.log_gamma[t])
        log_b -= logsumexp(self.log_gamma, axis=0)[:, np.newaxis]
        self.model.log_b = log_b

    def update_tr_params(self):
        self.log_ksi[:] = np.log(0.)
        log_a = np.log(np.zeros((self.n, self.n)))
        for t in range(self.T - 1):
            log_ksi_t = self.log_alpha[t][:, np.newaxis] + self.model.log_a \
                        + self.model.log_b[:, self.data[t + 1]] \
                        + self.log_beta[t + 1] - self._log_p

            log_a = np.logaddexp(log_a, log_ksi_t)

        log_sum_gamma = logsumexp(self.log_gamma, axis=0)[:, np.newaxis]
        self.model.log_a = log_a - log_sum_gamma
        self.model.log_pi = self.log_gamma[0]
