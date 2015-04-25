import numpy as np
import pylab as plt
from collections import defaultdict
from functools import reduce
from hmm_.hmm import HMMModel
from scipy.cluster.vq import kmeans2
from scipy.misc import logsumexp
from scipy.stats.mstats import mquantiles
from vlhmm_.context_tr_trie import ContextTransitionTrie
from vlhmm_.emission import GaussianEmission, PoissonEmission


class AbstractForwardBackward():
    def _init(self, data):
        self.T = len(data)
        self.data = data
        if data.ndim < 2:
            self.data = data[:, np.newaxis]
        self._init_a()
        self.n_contexts = self._get_n_contexts()
        self._init_auxiliary_params()
        self.track_log_p = defaultdict(list)
        self.track_e_params = {}

    def _init_auxiliary_params(self):
        self.log_alpha = np.log(np.zeros((self.T, self.n_contexts)))
        self.log_beta = np.log(np.zeros((self.T, self.n_contexts)))
        self.log_gamma = np.log(np.zeros((self.T, self.n_contexts)))
        self.log_ksi = np.log(np.zeros((self.T, self.n, self.n_contexts)))

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

    def _prepare_to_fitting(self, data, X=None, start="k-means", type_emission="Poisson", max_log_p_diff=1.5, **kwargs):
        self.start = start
        self.max_log_p_diff = max_log_p_diff
        if X is not None:
            self.start = "defined"
            self.X = X
        else:
            self._init_X(data, start)
        self._init(data)
        self._init_emission(type_emission)

    def fit(self, data, n_iter=150, X=None, log_pr_thresh=1e-2,
            start="k-means", type_emission="Poisson"):
        """
        :param data:
        :param n_iter:
        :param X: start hidden states
        :param log_pr_thresh: threshold for pruning
        :param start: {"k-means", "equal", "rand", "defined"}
        :param type_emission:
        :return:
        """
        self._prepare_to_fitting(data, X, start, type_emission)
        self._em(n_iter, log_pr_thresh)
        return self

    def _e_step(self):
            self.log_forward()
            self.log_backward()
            self._log_gamma()

    def _m_step(self):
        self.update_tr_params()
        self.update_emission_params()

    def _em(self, n_iter, log_pr_thresh):
        prev_log_p = np.log(0)
        for i in range(n_iter):
            self._e_step()
            self._m_step()
            print(self.emission.get_str_params(t_order="real"))
            print("_" * 75)
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

        self._log_p = logsumexp(self.log_alpha[-1])
        self.track_log_p[self.n_contexts].append(self._log_p)

    def log_backward(self):
        self.log_beta[:] = np.log(0.)
        self.log_beta[-1] = np.log(1.)
        for t in range(self.T - 2, -1, -1):
            for i in range(self.n_contexts):
                self._update_beta(t, i)

    def _log_gamma(self):
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
        self.log_ksi -=  \
            logsumexp(self.log_ksi, axis=(1, 2)).reshape((self.T, 1, 1))
        log_a = logsumexp(self.log_ksi[:-1], axis=0) -\
                logsumexp(self.log_gamma, axis=0)
        log_a -= logsumexp(log_a, axis=0)
        self.log_a = log_a
        print("a:\n{}".format(np.round(np.exp(self.log_a),4)))
        self._check_diff_log_p()

    def _check_diff_log_p(self, max_log_p_diff=None):
        if max_log_p_diff is None:
            max_log_p_diff = self.max_log_p_diff
        print("log_p {:.6}    ".format(self._log_p), end="")
        if len(self.track_log_p[self.n_contexts]) > 1:
            diff = self._log_p - self.track_log_p[self.n_contexts][-2]
            print("diff {:.4}".format(diff))
            if diff < 0:
                print("- " * 50)
            assert diff + max_log_p_diff > 0
        else:
            print()

    def plot_log_p(self):
        fig = plt.figure()

        l = len(self.track_log_p)
        min_y = min(map(lambda arr: mquantiles(arr, [1e-3])[0],
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
            if self.emission.name == "Poisson":
                ax.set_xlabel("{} iterations\n{}"
                              .format(num, self.track_e_params[n_context]))
            else:
                ax.set_xlabel("{} iterations".format(num))
            ax.plot(range(num), self.track_log_p[n_context], 'bo', range(num),
                    self.track_log_p[n_context], 'k')
            ax.set_ylim(min_y, max_y)
        return fig


class AbstractVLHMM():
    def __init__(self, n):
        """
        :param n: - length of context's alphabet, n <= 10
        :return:
        """
        self.n = n

    def _init_X(self, data, start):
        if start == "k-means":
            try:
                centre, labels = kmeans2(data, self.n)
            except TypeError:
                labels = np.random.choice(range(self.n), len(data))
        else:
            labels = np.random.choice(range(self.n), len(data))

        self.X = "".join(list(map(str, labels)))

    def get_list_c(self, *args):
        return self.tr_trie.get_list_c(
            reduce(lambda res, x: res + str(x), args, ""))


class VLHMMWang(AbstractVLHMM, AbstractForwardBackward):
    def _init(self, data):
        if self.start == "defined" or self.start == "k-means":
            print("X", self.X)
            self.tr_trie = ContextTransitionTrie(self.X, max_len=self.max_len,
                                                 n=self.n, start=self.start)
        else:
            self.tr_trie = \
                ContextTransitionTrie(None, max_len=self.max_len, n=self.n)
        print("_init..")
        self.contexts = self.tr_trie.seq_contexts
        self.state_c = np.array(list(
            map(lambda c: int(c[0]), self.contexts))).astype(np.int)
        self.n_contexts = self.tr_trie.n_contexts
        self.id_c = dict(zip(self.contexts, range(self.n_contexts)))
        self.log_context_p = np.log(np.ones(self.n_contexts) / self.n_contexts)

        super()._init(data)
        self.tr_trie.recount_with_log_a(self.log_a, self.contexts,
                                        self.log_context_p)
        print("n_contexts:", self.n_contexts)
        print("contexts: {}".format(self.contexts))
        print("a: \n{}".format(np.round(np.exp(self.log_a),2)))

    def _get_n_contexts(self):
        return self.n_contexts

    def _init_a(self):
        print(self.tr_trie.contexts.items())
        print(self.tr_trie.log_c_tr_trie.items())
        self.log_a = self.tr_trie.count_log_a(self.start)
        print(len(self.log_a))
        self.tr_trie.recount_with_log_a(self.log_a, self.contexts,
                                        self.log_context_p)

    def _prepare_to_fitting(self, *args, **kwargs):
        self.max_len = kwargs.get("max_len", 3)
        super()._prepare_to_fitting(*args, **kwargs)

    def fit(self, data, n_iter=75, th_prune=8e-3,
            log_pr_thresh=0.15, **kwargs):
        """
        :param data:
        :param n_iter:
        :param th_prune:
        :param log_pr_thresh:
        :param kwargs:
        max_len=4, X=None, start="k-means", type_emission="Poisson"
        :return:
        """
        self._prepare_to_fitting(data, th_prune=th_prune,
                                 log_pr_thresh=log_pr_thresh, **kwargs)
        changes = True
        while changes:
            self._em(n_iter, log_pr_thresh)
            changes = self._prune(th_prune)

        self.set_canonic_view()
        return self

    def _prune(self, th_prune):
        self.track_e_params[self.n_contexts] = self.emission.get_str_params()
        self.tr_trie.recount_with_log_a(self.log_a, self.contexts,
                                        self.log_context_p)
        prune = False
        while self.tr_trie.prune(th_prune):
            print("prune")
            print(self.tr_trie.seq_contexts)
            self.update_contexts()
            if self.n_contexts == 1:
                return False
            prune = True
        return prune

    def update_contexts(self):

        contexts = list(self.tr_trie.seq_contexts)
        n_contexts = len(contexts)
        log_context_p = np.log(np.zeros(n_contexts))
        id_c = dict(zip(contexts, range(n_contexts)))
        if n_contexts == 1:
            self.log_context_p[0] = logsumexp(self.log_context_p)
        else:
            for old_i, old_c in enumerate(self.contexts):
                assert len(self.tr_trie.get_list_c(old_c)) == 1, \
                    "old_c = {}, {}"\
                        .format(old_c, self.tr_trie.get_list_c(old_c))
                c =  self.tr_trie.get_c(old_c)
                i = id_c[c]
                log_context_p[i] = np.logaddexp(log_context_p[i],
                                         self.log_context_p[old_i])
            print("old_p: {}\n new_p: {}"
                  .format(np.round(np.exp(self.log_context_p), 2),
                          np.round(np.exp(log_context_p), 2)))
            self.log_context_p = log_context_p
        self.contexts = contexts
        self.n_contexts = n_contexts
        self.id_c = id_c
        self.log_a = self.tr_trie.count_log_a()
        if n_contexts == 1:
            return
        print("a\n{}".format(np.round(np.exp(self.log_a), 2)))
        self._init_auxiliary_params()
        self.state_c = np.array(list(
            map(lambda c: int(c[0]), self.contexts))).astype(np.int)
        print(self.state_c)
        assert [int(self.contexts[i][0]) for i in range(self.n_contexts)] \
               == [self.state_c[i] for i in range(self.n_contexts)], \
            "{}\n{}".format(
            [int(self.contexts[i][0]) for i in range(self.n_contexts)],
            [self.state_c[i] for i in range(self.n_contexts)])

    def _update_first_alpha(self, i):
        t = 0
        self.log_alpha[t][i] = \
            self.log_context_p[i] + \
            self.emission.log_p(self.data[t], self.state_c[i])

    def _update_next_alpha(self, t, i):
        self.log_alpha[t + 1][i] = np.log(0.)
        c = self.contexts[i]
        for c_ in self.tr_trie.get_list_c(c[1:]):
            i_ = self.id_c[c_]
            log_transition = self.log_a[self.state_c[i], i_]
            if len(c_) > t:
                log_transition = self.tr_trie.log_tr_p(c[0], c_[:t+1])

            self.log_alpha[t + 1][i] = np.logaddexp(
                self.log_alpha[t + 1][i],
                self.log_alpha[t][i_]
                + log_transition
                + self.emission.log_p(self.data[t + 1], self.state_c[i]))
    
    def _update_beta(self, t, i):
        self.log_beta[t][i] = np.log(0.)
        c = self.contexts[i]
        for q in range(self.n):
            for c_ in self.tr_trie.get_list_c(str(q) + c):
                i_ = self.id_c[c_]
                self.log_beta[t][i] = np.logaddexp(
                    self.log_beta[t][i],
                    self.log_a[q, i]
                    + self.emission.log_p(self.data[t + 1], q)
                    + self.log_beta[t + 1][i_])

    # def _log_gamma(self):
    #     super()._log_gamma()
    #     self.log_context_p = logsumexp(self.log_gamma, axis=0)
    #     self.log_context_p -= logsumexp(self.log_context_p)
    #     print("c_p = {}".format(np.round(np.exp(self.log_context_p),2)))

    def _update_ksi(self, t, q, i):
        self.log_ksi[t][q, i] = np.log(0.)
        for c_ in self.get_list_c(q, self.contexts[i]):
            i_ = self.id_c[c_]
            self.log_ksi[t][q, i] = \
                np.logaddexp(self.log_ksi[t][q, i],
                             self.log_alpha[t][i] +
                             self.log_a[q, i] +
                             self.emission.log_p(self.data[t+1], q) +
                             self.log_beta[t+1, i_])

    def update_tr_params(self):
        # print("alpha: \n", np.round(np.exp(self.log_alpha[:5]),2))
        # print("beta \n", np.round(np.exp(self.log_beta[:5]),2))
        # print("gamma \n", np.round(np.exp(self.log_gamma[:]),3))
        print("p_state ", np.round(np.exp(logsumexp(self.log_gamma, axis=0)), 4))
        self.log_context_p = logsumexp(self.log_gamma, axis=0)
        self.log_context_p -= logsumexp(self.log_context_p)
        print("c_p = {}".format(np.round(np.exp(self.log_context_p), 2)))
        super().update_tr_params()
        self.tr_trie.recount_with_log_a(self.log_a, self.contexts,
                                        self.log_context_p)

    def _get_log_gamma_emission(self):
        log_gamma_ = np.log(np.zeros((self.T, self.n)))
        for q in range(self.n):
            log_gamma_[:, q] = logsumexp(self.log_gamma[:, self.state_c==q],
                                         axis=1)
        log_gamma_ -= logsumexp(log_gamma_, axis=1)[:, np.newaxis]
        return log_gamma_

    def update_emission_params(self):
        self.emission.update_params(self._get_log_gamma_emission())

    @staticmethod
    def _get_context_order(contexts, state_order):
        n = len(state_order)
        d = dict(zip(state_order, range(n)))
        new_contexts = []
        for c in contexts:
            new_c = "".join(list(map(lambda q: str(d[int(q)]), c)))
            new_contexts.append(new_c)
        tmp = sorted(zip(new_contexts, range(len(contexts))))
        c_order = list(map(lambda x: x[1], tmp))
        return c_order

    @staticmethod
    def get_sorted_contexts_and_log_a(contexts, log_a, state_order):
        c_order= VLHMMWang._get_context_order(contexts, state_order)

        return list(np.array(contexts)[c_order]), log_a[state_order, :][:, c_order]

    def set_canonic_view(self):
        state_order = self.emission.get_order()
        c_order = self._get_context_order(self.contexts, state_order)
        contexts = list(map(lambda i: self.contexts[c_order[i]],
                            range(self.n_contexts)))
        self.contexts = contexts
        self.log_a = self.log_a[state_order, :][:, c_order]
        self.log_context_p = self.log_context_p[c_order]
        self.tr_trie.recount_with_log_a(self.log_a, self.contexts,
                                        self.log_context_p)
        self.emission.set_canonic_view()

    def sample(self, size, start=0):
        X = np.zeros((size, self.emission.n))
        c = self.contexts[start]
        index_c = self.id_c[c]
        states = np.arange(self.n)
        for i in range(size):
            q = int(np.random.choice(states, p=np.exp(self.log_a[:, index_c])))
            X[i] = self.emission.sample(q)
            c = self.tr_trie.get_c(str(q)+c)
            index_c = self.id_c[c]
        return X


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
                labels = np.random.choice(range(self.n), len(data))
        else:
            labels = np.random.choice(range(self.n), len(data))
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
        print("a:\n{}".format(np.round(np.exp(self.model.log_a),2)))
        print("log_p {:.3}".format(self._log_p))


