import numpy as np
from scipy.misc import logsumexp
from hmm_.hmm import HMMModel
from vlhmm_.vlhmm import ContextTrie, VLHMM, GaussianEmission


class AbstractForwardBackward():
    def _init(self, data):
        self.T = len(data)
        self.data = data
        self._init_a()
        self._init_emission()
        self.n_states = self._get_n_states()
        self.log_alpha = np.zeros((self.T, self.n_states))
        self.log_beta = np.zeros((self.T, self.n_states))
        self.log_gamma = np.zeros((self.T, self.n_states))
        self.ksi = np.zeros((self.T, self.n, self.n_states))

    def _init_X(self, data):
        pass

    def _get_n_states(self):
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

        for i in range(self.n_states):
            self._update_first_alpha(i)
        for t in range(self.T - 1):
            for i in range(self.n_states):
                self._update_next_alpha(t, i)

    def log_backward(self):
        self.log_beta[:] = np.log(0.)
        self.log_beta[-1] = np.log(1.)
        for t in range(self.T - 2, -1, -1):
            for i in range(self.n_states):
                self._update_beta(t, i)

    def _log_gamma(self):
        self._log_p = logsumexp(self.log_alpha[-1])
        self.log_gamma = self.log_alpha + self.log_beta - self._log_p

    def update_emission_params(self):
        self.emission.update_params(self.log_gamma)

    def update_tr_params(self):
        self.ksi[:] = np.log(0.)
        for t in range(self.T - 1):
            for i in range(self.n_states):
                for q in range(self.n):
                    self._update_ksi(t, q, i)

        for i in range(self.n_states):
            sum_gamma = logsumexp(self.log_gamma[:, i])
            for q in range(self.n):
                self.log_a[q, i] = logsumexp(self.ksi[:, q, i]) - sum_gamma

        self.log_a = self.log_a - logsumexp(self.log_a,
                                            axis=0)  # strange normalisation..
        print("a:\n{}".format(np.exp(self.log_a)))
        print("log_p {}".format(self._log_p))



class VLHMMWang(VLHMM, AbstractForwardBackward):
    def _init(self, data):
        self.context_trie = ContextTrie(self.X, max_len=self.max_len)
        self.contexts = list(self.context_trie.contexts)
        self.n_contexts = len(self.contexts)
        self.data_contexts = list(self.context_trie.get_contexts(self.X))
        self.id_c = dict(zip(self.contexts, range(self.n_contexts)))
        log_context_p = np.array(
            [self.context_trie.log_p(c) for c in self.contexts])
        self.log_context_p = log_context_p - logsumexp(log_context_p)

        super()._init(data)

        len_print = 4
        print("X: {}".format(self.X))
        print("n_contexts:", self.n_contexts)
        print("contexts: {}".format(self.contexts))
        print("data_contexts: {}".format(self.data_contexts))
        print("a: \n{}".format(np.exp(self.log_a)))
        print("context_p: {}".format(np.exp(self.log_context_p)))
        print("init_mu = {}..\ninit_sigma = {}..\n".format(
            self.emission.mu[:len_print], self.emission.sigma[:len_print]))

    def _init_emission(self):
        labels = list(map(lambda c: self.id_c[c], self.data_contexts))
        self.emission = GaussianEmission(self.data, labels, self.n_contexts)

    def _get_n_states(self):
        return self.n_contexts

    def _init_a(self):
        self.log_a = np.array(
            [[self.context_trie.log_tr_p(str(q), self.contexts[l])
              for l in range(self.n_contexts)]
             for q in range(self.n)])

    def fit(self, data,  max_len=4, X=None, n_iter=5, th_prune=1e-4):
        self.max_len=max_len
        super().fit(data, n_iter, X)
        self._prune(th_prune)
        return self


    def _prune(self, th_prune):
        changes = self.context_trie.prune(th_prune)
        while changes > 0:
            print("prune", changes)
            changes = self.context_trie.prune(th_prune)

    def _update_first_alpha(self, i):
        self.log_alpha[0][i] = self.log_context_p[i] \
                              + self.emission.log_p(self.data[0], i)

    def _update_next_alpha(self, t, i):
        c = self.contexts[i]
        for q in range(self.n):
            c_ = self.get_c(c[1:], q)
            j = self.id_c[c_]
            self.log_alpha[t + 1][i] = np.logaddexp(
                self.log_alpha[t + 1][i],
                self.log_alpha[t][j]
                + self.log_a[int(c[0]), j]
                + self.emission.log_p(self.data[t + 1], i))

    def _update_beta(self, t, i):
        c = self.contexts[i]
        for q in range(self.n):
            c_ = self.get_c(q, c)
            j = self.id_c[c_]
            self.log_beta[t][i] = np.logaddexp(
                self.log_beta[t][i],
                self.log_a[q, i]
                + self.emission.log_p(self.data[t + 1], j)
                + self.log_beta[t + 1][j])

    def _update_ksi(self, t, q, i):
        i_ = self.id_c[self.get_c(q, self.contexts[i])]
        self.ksi[t][q, i] = self.log_alpha[t][i] + self.log_a[q, i]
        + self.emission.log_p(self.data[t + 1], i_) \
        + self.log_beta[t + 1][i_] - self._log_p


class HMM(AbstractForwardBackward):
    def __init__(self, n):
        self.n = n

    def _init(self, data):
        self.m = len(set(data))
        self.model = HMMModel.get_random_model(self.n, self.m)
        super()._init(data)

    def _get_n_states(self):
        return self.n

    def update_emission_params(self):
        log_b = np.log(np.zeros((self.n, self.m)))
        for t in range(self.T - 1):
            log_b[:, self.data[t]] = np.logaddexp(log_b[:, self.data[t]], self.log_gamma[t])
        log_b -= logsumexp(self.log_gamma, axis=0)[:, np.newaxis]
        self.model.log_b = log_b

    def _update_first_alpha(self, i):
        self.log_alpha[0][i] = self.model.log_pi[i] + self.model.log_b[i, self.data[0]]

    def _update_next_alpha(self, t, j):
        tmp = logsumexp(self.log_alpha[t] + self.model.log_a[:, j])
        self.log_alpha[t+1][j] = tmp + self.model.log_b[j, self.data[t+1]]

    def _update_beta(self, t, i):
        self.log_beta[t][i] = logsumexp(
                self.model.log_a[i] + (self.model.log_b[:, self.data[t + 1]] + self.log_beta[t + 1]))

    def update_tr_params(self):
        self.ksi[:] = np.log(0.)
        log_a = np.log(np.zeros((self.n, self.n)))
        for t in range(self.T - 1):
            log_ksi_t = self.log_alpha[t][:, np.newaxis] + self.model.log_a \
                        + self.model.log_b[:, self.data[t + 1]] \
                        + self.log_beta[t + 1] - self._log_p

            log_a = np.logaddexp(log_a, log_ksi_t)

        log_sum_gamma = logsumexp(self.log_gamma, axis=0)[:, np.newaxis]
        self.model.log_a = log_a - log_sum_gamma
        self.model.log_pi = self.log_gamma[0]
