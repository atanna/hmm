import numpy as np
from scipy.misc import logsumexp
from vlhmm_.vlhmm import VLHMM, GaussianEmission, ContextTrie


len_print = 4


class VLHMMWang(VLHMM):
    def _init(self, data, max_len):
        self.T = len(data)
        self.data = data
        self._init_X(data)
        self.context_trie = ContextTrie(self.X, max_len=max_len)
        self.contexts = list(self.context_trie.contexts)
        self.n_contexts = len(self.contexts)
        self.data_contexts = list(self.context_trie.get_contexts(self.X))
        self.id_c = dict(zip(self.contexts, range(self.n_contexts)))
        self._init_a()
        log_context_p = np.array(
            [self.context_trie.log_p(c) for c in self.contexts])
        self.log_context_p = log_context_p - logsumexp(log_context_p)
        labels = list(map(lambda c: self.id_c[c], self.data_contexts))
        self.emission = GaussianEmission(data, labels, self.n_contexts)

        print("X: {}".format(self.X))
        print("n_contexts:", self.n_contexts)
        print("contexts: {}".format(self.contexts))
        print("data_contexts: {}".format(self.data_contexts))
        print("a: \n{}".format(np.exp(self.log_a)))
        print("context_p: {}".format(np.exp(self.log_context_p)))
        print("init_mu = {}..\ninit_sigma = {}..\n".format(
            self.emission.mu[:len_print], self.emission.sigma[:len_print]))

    def _init_a(self):
        self.log_a = np.array(
            [[self.context_trie.log_tr_p(str(q), self.contexts[l])
              for l in range(self.n_contexts)]
             for q in range(self.n)])

    def fit(self, data, max_len=4, n_iter=5, th_prune=1e-4):
        self._init(data, max_len)
        self._em(n_iter)
        self._prune(th_prune)
        return self.X                   #fix it

    def _em(self, n_iter):
        def e_step():
            log_alpha = self.log_forward()
            log_beta = self.log_backward()
            log_gamma = self._log_gamma(log_alpha, log_beta)

            print("_alpha: {}..".format(np.exp(log_alpha[:len_print])))
            print("_beta: {}..".format(np.exp(log_beta[:len_print])))
            print("_gamma: {}..".format(np.exp(log_gamma[:len_print])))
            print("-" * 10, "log_p={},  p={}"
                  .format(self._log_p, np.exp(self._log_p)))

            return log_alpha, log_beta, log_gamma

        def m_step(log_alpha, log_beta, log_gamma):
            self.update_emission_params(log_gamma)
            self.update_tr_params(log_alpha, log_beta, log_gamma)

        for i in range(n_iter):
            m_step(*e_step())

    def _prune(self, th_prune):
        changes = self.context_trie.prune(th_prune)
        while changes > 0:
            print("prune", changes)
            changes = self.context_trie.prune(th_prune)

    def log_forward(self):
        log_alpha = np.zeros((self.T, self.n_contexts))
        log_alpha[:] = -np.inf

        for l in range(self.n_contexts):
            log_alpha[0][l] = self.log_context_p[l] \
                              + self.emission.log_p(self.data[0], l)

        for t in range(self.T - 1):
            for l in range(self.n_contexts):
                c = self.contexts[l]
                for q in range(self.n):
                    c_ = self.get_c(c[1:], q)
                    l_ = self.id_c[c_]
                    log_alpha[t + 1][l] = np.logaddexp(
                        log_alpha[t + 1][l],
                        log_alpha[t][l_]
                        + self.log_a[int(c[0]), l_]
                        + self.emission.log_p(self.data[t + 1], l))
        return log_alpha

    def log_backward(self):
        log_beta = np.zeros((self.T, self.n_contexts))
        log_beta[:] = np.log(0.)
        log_beta[-1] = np.log(1.)
        for t in range(self.T - 2, -1, -1):
            for l in range(self.n_contexts):
                c = self.contexts[l]
                for q in range(self.n):
                    c_ = self.get_c(q, c)
                    l_ = self.id_c[c_]
                    log_beta[t][l] = np.logaddexp(
                        log_beta[t][l],
                        self.log_a[q, l]
                        + self.emission.log_p(self.data[t + 1], l_)
                        + log_beta[t + 1][l_])
        return log_beta

    def _log_gamma(self, log_alpha, log_beta):
        self._log_p = logsumexp(log_alpha[-1])
        return log_alpha + log_beta - self._log_p

    def update_emission_params(self, log_gamma):
        self.emission.update_params(log_gamma)

        print("emission: \nmu {}..\nsigma {}..".format(
            self.emission.mu[:len_print], self.emission.sigma[:len_print]))

    def update_tr_params(self, log_alpha, log_beta, log_gamma):
        self.ksi = np.zeros((self.T, self.n, self.n_contexts))
        self.ksi[:] = np.log(0.)
        for t in range(self.T - 1):
            for l in range(self.n_contexts):
                for q in range(self.n):
                    l_ = self.id_c[self.get_c(q, self.contexts[l])]
                    self.ksi[t][q, l] = log_alpha[t][l] + self.log_a[q, l]
                    + self.emission.log_p(self.data[t + 1], l_) \
                    + log_beta[t + 1][l_] - self._log_p

        for l in range(self.n_contexts):
            sum_gamma = logsumexp(log_gamma[:, l])
            for q in range(self.n):
                self.log_a[q, l] = logsumexp(self.ksi[:, q, l]) - sum_gamma

        self.log_a = self.log_a - logsumexp(self.log_a,
                                            axis=0)  # strange normalisation..
        print("a:\n{}".format(np.exp(self.log_a)))


