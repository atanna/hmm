import numpy as np
import pylab as plt
from collections import defaultdict
from scipy import stats
from scipy.cluster.vq import kmeans2
from scipy.misc import logsumexp
from scipy.stats.mstats import mquantiles
from hmm_.hmm import HMMModel
from vlhmm_.context_tr_trie import ContextTransitionTrie
from vlhmm_.vlhmm import AbstractVLHMM, GaussianEmission, PoissonEmission


class AbstractForwardBackward():
    def _init(self, data):
        self.T = len(data)
        self.data = data
        if data.ndim < 2:
            self.data = data[:, np.newaxis]
        self._init_a()
        self.n_contexts = self._get_n_contexts()
        self.log_alpha = np.zeros((self.T, self.n_contexts))
        self.log_beta = np.zeros((self.T, self.n_contexts))
        self.log_gamma = np.zeros((self.T, self.n_contexts))
        self.log_ksi = np.zeros((self.T, self.n, self.n_contexts))
        self.track_log_p = defaultdict(list)

    def _init_X(self, data, start):
        pass

    def _get_n_contexts(self):
        pass

    def _init_a(self):
        pass

    def _init_emission(self, type_emission):
        if type_emission == "Poisson":
            self.emission = PoissonEmission(self.data, self.X, self.n)
        else:
            self.emission = GaussianEmission(self.data, self.X, self.n)
            print("init_mu = {}..\ninit_sigma = {}..\n".format(
                self.emission.mu, self.emission.sigma))

    def fit(self, data, n_iter=150, X=None, log_pr_thresh=1e-2, start="k-means", type_emission="Poisson"):
        """
        :param data:
        :param n_iter:
        :param X: start hidden states
        :param log_pr_thresh: threshold for pruning
        :param start: {"k-means", "equal", "rand"}
        :param type_emission:
        :return:
        """
        self.start = start
        if X is not None:
            self.X = X
        else:
            self._init_X(data, start)
        self._init(data)
        self._init_emission(type_emission)
        self._em(n_iter, log_pr_thresh)
        return self

    def _em(self, n_iter, log_pr_thresh):
        def e_step():
            self.log_forward()
            self.log_backward()
            self._log_gamma()

        def m_step():
            self.update_tr_params()
            self.update_emission_params()
        prev_log_p = np.log(0)
        for i in range(n_iter):
            e_step()
            m_step()
            self.track_log_p[self.n_contexts].append(self._log_p)
            if np.abs(prev_log_p - self._log_p) < log_pr_thresh:
                return
            prev_log_p = self._log_p

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
        self.log_ksi -= \
            logsumexp(self.log_ksi, axis=(1, 2)).reshape((self.T, 1, 1))
        # log_gamma = logsumexp(self.log_ksi[:-1], axis=1)
        # print(np.exp(logsumexp(self.log_ksi[:-1], axis=0) -\
        #  logsumexp(log_gamma, axis=0)))
        log_a = logsumexp(self.log_ksi[:-1], axis=0) -\
                logsumexp(self.log_gamma, axis=0)
        log_a -= logsumexp(log_a, axis=0)
        self.log_a = log_a
        print("a:\n{}".format(np.exp(self.log_a)))
        print("log_p {}".format(self._log_p))

    def plot_log_p(self):
        fig = plt.figure()

        l = len(self.track_log_p)
        min_y = min(map(lambda arr: mquantiles(arr, [0.01])[0],
                        self.track_log_p.values()))
        max_y = max(map(max, self.track_log_p.values()))
        dy = (max_y-min_y)/8
        min_y = int(min_y - dy)
        max_y = int(max_y + dy)

        keys = sorted(self.track_log_p.keys(), reverse=True)

        for i, n_context in enumerate(keys):
            ax = fig.add_subplot(1, l, i+1)
            ax.set_title("n_contexts = {}".format(n_context))
            if i == 0:
                ax.set_ylabel('log_p')
            num = len(self.track_log_p[n_context])
            ax.set_xlabel("{} iterations".format(num))
            ax.plot(range(num), self.track_log_p[n_context], 'bo', range(num),
                    self.track_log_p[n_context], 'k')
            ax.set_ylim(min_y, max_y)
        return fig


class VLHMMWang(AbstractVLHMM, AbstractForwardBackward):
    def _init(self, data):
        if self.start != "k-means":
            self.tr_trie = \
                ContextTransitionTrie(None, max_len=self.max_len, n=self.n)
        else:
            self.tr_trie = ContextTransitionTrie(self.X, max_len=self.max_len)

        self.contexts = self.tr_trie.seq_contexts
        self.state_c = list(map(lambda c: int(c[0]), self.contexts))
        self.n_contexts = self.tr_trie.n_contexts
        self.id_c = dict(zip(self.contexts, range(self.n_contexts)))
        self.log_context_p = np.log(np.ones(self.n_contexts) / self.n_contexts)

        super()._init(data)
        self.tr_trie.recount_with_log_a(self.log_a, self.contexts)
        print("n_contexts:", self.n_contexts)
        print("contexts: {}".format(self.contexts))
        print("a: \n{}".format(np.exp(self.log_a)))

    def _get_n_contexts(self):
        return self.n_contexts

    def _init_a(self):
        self.log_a = self.tr_trie.count_log_a(self.start)
        print(len(self.log_a))
        self.tr_trie.recount_with_log_a(self.log_a, self.contexts)

    def fit(self, data, max_len=4, n_iter=55, th_prune=1e-2, log_pr_thresh=1e-2, **kwargs):
        self.max_len = max_len
        super().fit(data, n_iter, log_pr_thresh=log_pr_thresh, **kwargs)
        self.tr_trie.recount_with_log_a(self.log_a, self.contexts)
        while self._prune(th_prune):
            self._em(n_iter, log_pr_thresh)
            self.tr_trie.recount_with_log_a(self.log_a, self.contexts)
        return self

    @staticmethod
    def get_sorted_contexts_and_log_a(contexts, log_a, order):
        n = len(log_a)
        d = dict(zip(order, range(n)))
        new_contexts =  []
        for c in contexts:
            new_c = "".join(list(map(lambda q: str(d[int(q)]), c)))
            new_contexts.append(new_c)
        new_log_a = log_a[order, :]

        tmp = sorted(zip(new_contexts, range(len(contexts))))
        new_contexts = list(map(lambda x: x[0], tmp))
        order = list(map(lambda x: x[1], tmp))
        new_log_a = new_log_a[:,order]

        return new_contexts, new_log_a

    def sample(self, size, start=0):
        X = np.zeros((size, self.emission.n))
        c = self.contexts[start]
        index_c = self.id_c[c]
        states = np.arange(self.n)
        for i in range(size):
            q = stats.rv_discrete(name='custm',
                                  values=(states,
                                          np.exp(
                                              self.log_a[:, index_c]))).rvs()
            X[i] = self.emission.sample(q)
            c = self.get_c(q, c)
            index_c = self.id_c[c]
        return X

    def _prune(self, th_prune):
        prune = changes = self.tr_trie.prune(th_prune)
        while changes:
            print("prune", changes)
            print(self.tr_trie.seq_contexts)
            self.update_contexts()
            if self.n_contexts == 1:
                return False
            changes = self.tr_trie.prune(th_prune)
        return prune

    def update_contexts(self):
        self.contexts = list(self.tr_trie.seq_contexts)
        self.n_contexts = len(self.contexts)
        self.id_c = dict(zip(self.contexts, range(self.n_contexts)))
        self.log_a = self.tr_trie.count_log_a()
        if self.n_contexts > 1:
            self.state_c = list(map(lambda c: int(c[0]), self.contexts))
        self.log_alpha = np.zeros((self.T, self.n_contexts))
        self.log_beta = np.zeros((self.T, self.n_contexts))
        self.log_gamma = np.zeros((self.T, self.n_contexts))
        self.log_ksi = np.zeros((self.T, self.n, self.n_contexts))
        self.log_context_p = np.log(np.ones(self.n_contexts) / self.n_contexts)

    def _update_first_alpha(self, i):
        self.log_alpha[0][i] = \
            self.log_context_p[i] + \
            self.emission.log_p(self.data[0], self.state_c[i])

    def _update_next_alpha(self, t, i):
        self.log_alpha[t + 1][i] = np.log(0.)
        c = self.contexts[i]
        for q in range(self.n):
            for c_ in self.tr_trie.get_list_c(c[1:]+str(q)):
                i_ = self.id_c[c_]
                if len(c_) > t:
                    c_ = c_[:t]
                    log_transition = self.tr_trie.log_tr_p(c[0], c_)
                else:
                    log_transition = self.log_a[int(c[0]), i_]

                self.log_alpha[t + 1][i] = np.logaddexp(
                    self.log_alpha[t + 1][i],
                    self.log_alpha[t][i_]
                    + log_transition
                    + self.emission.log_p(self.data[t + 1], self.state_c[i]))
    
    def _update_beta(self, t, i):
        self.log_beta[t][i] = np.log(0.)
        c = self.contexts[i]
        for q in range(self.n):
            c_ = self.get_c(q, c)
            i_ = self.id_c[c_]
            self.log_beta[t][i] = np.logaddexp(
                self.log_beta[t][i],
                self.log_a[q, i]
                + self.emission.log_p(self.data[t + 1], self.state_c[i_])
                + self.log_beta[t + 1][i_])

    def _log_gamma(self):
        super()._log_gamma()
        self.log_context_p = logsumexp(self.log_gamma, axis=0)
        self.log_context_p -= logsumexp(self.log_context_p)

    def _update_ksi(self, t, q, i):

        self.log_ksi[t][q, i] = self.log_alpha[t][i] + \
                                self.log_a[q,i] + \
                                self.emission.log_p(self.data[t+1], q) + \
                                self.log_beta[t+1, q]

    def update_tr_params(self):
        super().update_tr_params()
        self.tr_trie.recount_with_log_a(self.log_a, self.contexts)

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
        super()._init(data)

    def _init_X(self, data, start):
        if start == "k-means":
            try:
                centre, labels = kmeans2(data, self.n)
            except TypeError:
                labels= np.random.choice(range(self.n), len(data))
        else:
            labels= np.random.choice(range(self.n), len(data))
        self.X = "".join(list(map(str, labels)))

    def _init_a(self):
        if self.start == "start":
            self.log_a = np.log(np.ones((self.n, self.n)) / self.n)
            self.log_pi = np.log(np.ones(self.n) / self.n)
            return
        model = HMMModel.get_random_model(self.n, 1)
        self.log_pi = model.log_pi
        self.log_a = (model.log_a).T

    def _get_n_contexts(self):
        return self.n

    def _update_first_alpha(self, i):
        self.log_alpha[0][i] = self.log_pi[i] +\
                               self.emission.log_p(self.data[0], i)

    def _update_next_alpha(self, t, i):
        self.log_alpha[t+1, i] = \
            logsumexp(self.log_alpha[t] + self.log_a[i, :]) +\
            self.emission.log_p(self.data[t+1], i)

    def _update_beta(self, t, i):
        self.log_beta[t][i] = logsumexp(
            self.log_a[:, i] +
            self.emission.all_log_p(self.data[t + 1]) +
            self.log_beta[t + 1])

    def _update_ksi(self, t, q, i):
        self.log_ksi[t][q, i] = self.log_alpha[t][i] + \
                                self.log_a[q,i] + \
                                self.emission.log_p(self.data[t+1], q) + \
                                self.log_beta[t+1, q]

    def update_emission_params(self):
        self.emission.update_params(self.log_gamma)


class DiscreteHMM(AbstractForwardBackward):
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
            log_b[:, self.data[t][0]] = np.logaddexp(log_b[:, self.data[t][0]],
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

        log_a -= log_sum_gamma
        log_a -= logsumexp(log_a, axis=0)
        self.model.log_a = log_a
        self.model.log_pi = self.log_gamma[0]
        # print("gamma", np.exp(self.log_gamma))
        print("a:\n{}".format(np.exp(self.model.log_a)))
        print("log_p {}".format(self._log_p))
