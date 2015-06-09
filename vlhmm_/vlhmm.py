import numpy as np
import pylab as plt
from collections import defaultdict
from functools import reduce
from scipy.cluster.vq import kmeans2
from scipy.misc import logsumexp
from scipy.stats.mstats import mquantiles
from context_tr_trie import ContextTransitionTrie
from emission import GaussianEmission, PoissonEmission
import time
import _vlhmmc as _vlhmmc


class AbstractForwardBackward():
    def _init(self, data, **kwargs):
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

    def _init_emission(self, type_emission, start_params=None):
        if type_emission == "Poisson":
            self.data = self.data.astype(np.int32)
            if start_params is not None:
                self.emission = PoissonEmission(n_states=self.n) \
                    ._set_params(start_params["alpha"])
            else:
                self.emission = PoissonEmission(self.data, self.X, self.n)
        else:
            self.emission = GaussianEmission(self.data, self.X, self.n)
            if self._print:
                print("init_mu = {}..\ninit_sigma = {}..\n".format(
                    self.emission.mu, self.emission.sigma))
        self.log_b = self.emission.get_log_b(self.data)

    def _prepare_to_fitting(self, data, X=None, start="k-means",
                            type_emission="Poisson", max_log_p_diff=1.5,
                            _print=False,
                            **kwargs):
        self.start = start
        self.max_log_p_diff = max_log_p_diff
        self._print = _print

        if X is not None:
            self.start = "defined"
            self.X = X
        else:
            self._init_X(data, start)
        self._init(data, **kwargs)
        self._init_emission(type_emission)

    def fit(self, data, n_iter=150, X=None, log_pr_thresh=0.15,
            start="k-means", type_emission="Poisson"):
        """
        :param data:
        :param n_iter:
        :param X: start hidden states
        :param log_pr_thresh: threshold for em
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
        self._log_ksi()

    def _m_step(self):
        self.update_tr_params()
        self.update_emission_params()

    def _em(self, n_iter, log_pr_thresh):
        prev_log_p = np.log(0)
        for i in range(n_iter):
            self._e_step()
            self._m_step()
            if self._print:
                print(self.emission.get_str_params(t_order="real"))
                print("_" * 75)
            if self._log_p - prev_log_p < log_pr_thresh:
                return
            prev_log_p = self._log_p

    def _update_ksi(self, t, q, i):
        pass

    def log_forward(self):
        pass

    def log_backward(self):
        pass

    def _log_gamma(self):
        self.log_gamma = self.log_alpha + self.log_beta
        self.log_gamma -= logsumexp(self.log_gamma, axis=1)[:, np.newaxis]

    def _log_ksi(self):
        pass

    def _normalise_a(self):
        self.log_a = self.log_a - logsumexp(self.log_a, axis=0)

    def update_emission_params(self):
        self.emission.update_params(self.log_gamma)
        self.log_b = self.emission.get_log_b(self.data)

    def update_tr_params(self):
        pass

    def _check_diff_log_p(self, max_log_p_diff=None):
        if max_log_p_diff is None:
            max_log_p_diff = self.max_log_p_diff
        if self._print:
            print("log_p {:.6}    ".format(self._log_p), end="")
        if len(self.track_log_p[self.n_contexts]) > 1:
            diff = self._log_p - self.track_log_p[self.n_contexts][-2]
            if self._print:
                print("diff {:.4}".format(diff))
                if diff < 0:
                    print("- " * 50)
            assert diff + max_log_p_diff > 0
        else:
            if self._print:
                print()

    def plot_log_p(self, language="en"):
        if language == "ru":
            s_log_p = "Логарифм правдоподобия"
            s_iterations = "итераций"
            s_n_contexts = "Число контекстов"
        else:
            s_log_p = 'log_p'
            s_iterations = 'iterations'
            s_n_contexts = 'n_contexts'
        fig = plt.figure()

        l = len(self.track_log_p)
        min_y = min(map(lambda arr: mquantiles(arr, [1e-3])[0],
                        self.track_log_p.values()))
        max_y = max(map(max, self.track_log_p.values()))
        dy = (max_y - min_y) / 8
        min_y = int(min_y - dy)
        max_y = int(max_y + dy)

        keys = sorted(self.track_log_p.keys(), reverse=True)

        for i, n_context in enumerate(keys):
            ax = fig.add_subplot(1, l, i + 1)
            ax.set_title("{} = {}".format(s_n_contexts, n_context))
            if i == 0:
                ax.set_ylabel(s_log_p)
            num = len(self.track_log_p[n_context])
            if self.emission.name == "Poisson":
                ax.set_xlabel("{} {}\n{}"
                              .format(num, s_iterations,
                                      self.track_e_params[n_context]))
            else:
                ax.set_xlabel("{} {}".format(s_iterations, num))
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
        elif start == "equal":
            labels = np.random.choice(range(self.n), len(data))

        self.X = "".join(list(map(str, labels)))

    def get_list_c(self, *args):
        return self.tr_trie.get_list_c(
            reduce(lambda res, x: res + str(x), args, ""))


class VLHMM(AbstractVLHMM, AbstractForwardBackward):
    def _init(self, data, start_params=None, **kwargs):
        self.log_context_p = None
        if start_params is not None:
            self.tr_trie = \
                ContextTransitionTrie(n=self.n, max_len=self.max_len) \
                    .recount_with_log_a(start_params["log_a"],
                                        start_params["contexts"],
                                        start_params["log_c_p"])
        elif self.start == "defined" or self.start == "k-means":
            self.tr_trie = ContextTransitionTrie(self.X, max_len=self.max_len,
                                                 n=self.n, start=self.start)
        else:
            self.tr_trie = \
                ContextTransitionTrie(None, max_len=self.max_len, n=self.n)
        self.contexts = self.tr_trie.seq_contexts
        self.state_c = np.array(list(
            map(lambda c: int(c[0]), self.contexts))).astype(np.uint8)
        self.n_contexts = self.tr_trie.n_contexts
        self.id_c = dict(zip(self.contexts, range(self.n_contexts)))
        if self.log_context_p is None:
            self.log_context_p = np.log(np.ones(self.n_contexts)
                                        / self.n_contexts)
        self.info = []

        super()._init(data, **kwargs)

        self.tr_trie.recount_with_log_a(self.log_a, self.contexts,
                                        self.log_context_p)
        self.mask = self.get_context_mask()
        if self._print:
            print("n_contexts:", self.n_contexts)
            print("contexts: {}".format(self.contexts))
            print("a: \n{}".format(np.round(np.exp(self.log_a), 2)))

    def _get_n_contexts(self):
        return self.n_contexts

    def _init_a(self):
        self.log_a = self.tr_trie.count_log_a(self.start)
        self.tr_trie.recount_with_log_a(self.log_a, self.contexts,
                                        self.log_context_p)

    def _prepare_to_fitting(self, *args, **kwargs):
        self.max_len = kwargs.get("max_len", 3)
        super()._prepare_to_fitting(*args, **kwargs)

    def fit(self, data, n_iter=75, th_prune=0.007, log_pr_thresh=0.05,
            **kwargs):
        """
        :param data:
        :param n_iter:
        :param th_prune: thresholf for prunning
        :param log_pr_thresh: threshold for em
        :param kwargs:
        max_len=4, X=None, start="k-means", type_emission="Poisson",
        _print=False
        :return:
        """
        time_start = time.time()
        self._prepare_to_fitting(data, th_prune=th_prune,
                                 log_pr_thresh=log_pr_thresh, **kwargs)
        changes = True
        while changes:
            self._em(n_iter, log_pr_thresh)
            changes = self._prune(th_prune)

        self.set_canonic_view()
        self.fit_time = time.time() - time_start
        return self

    def _prune(self, th_prune):
        info = "{} contexts: {}\nc_p: {}\na: {}\n{}\nlog_p: {}\naic: {}\n" \
            .format(self.n_contexts, self.contexts,
                    np.round(np.exp(self.log_context_p), 3),
                    np.round(np.exp(self.log_a), 2),
                    self.emission.get_str_params(),
                    self._log_p, self.get_aic())
        self.info.append(info)

        self.track_e_params[self.n_contexts] = self.emission.get_str_params()
        self.tr_trie.recount_with_log_a(self.log_a, self.contexts,
                                        self.log_context_p)
        prune = False
        while self.tr_trie.prune(th_prune):
            if self._print:
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
                    "old_c = {}, {}" \
                        .format(old_c, self.tr_trie.get_list_c(old_c))
                c = self.tr_trie.get_c(old_c)
                i = id_c[c]
                log_context_p[i] = np.logaddexp(log_context_p[i],
                                                self.log_context_p[old_i])
            if self._print:
                print("old_p: {}\n new_p: {}"
                      .format(np.round(np.exp(self.log_context_p), 2),
                              np.round(np.exp(log_context_p), 2)))
            self.log_context_p = log_context_p
        self.contexts = contexts
        self.n_contexts = n_contexts
        self.id_c = id_c
        self.log_a = self.tr_trie.count_log_a()
        self.mask = self.get_context_mask()
        if n_contexts == 1:
            return
        self._init_auxiliary_params()
        self.state_c = np.array(list(
            map(lambda c: int(c[0]), self.contexts))).astype(np.uint8)

        if self._print:
            print("a\n{}".format(np.round(np.exp(self.log_a), 2)))
            print(self.state_c)
        assert [int(self.contexts[i][0]) for i in range(self.n_contexts)] \
               == [self.state_c[i] for i in range(self.n_contexts)], \
            "{}\n{}".format(
                [int(self.contexts[i][0]) for i in range(self.n_contexts)],
                [self.state_c[i] for i in range(self.n_contexts)])

    def log_forward(self):
        _vlhmmc._log_forward(self.mask,
                             self.log_a, self.log_b,
                             self.log_context_p,
                             self.state_c, self.log_alpha)
        self._log_p = logsumexp(self.log_alpha[-1])
        self.track_log_p[self.n_contexts].append(self._log_p)

    def log_backward(self):
        _vlhmmc._log_backward(self.mask, self.log_a, self.log_b, self.log_beta)

    def _log_ksi(self):
        _vlhmmc._log_ksi(self.mask,
                         self.log_a, self.log_b, self.log_alpha,
                         self.log_beta, self.log_ksi)

    def get_context_mask(self):
        mask = np.zeros((self.n, self.n_contexts, self.n_contexts)).astype(
            np.uint8)
        for q in range(self.n):
            for i, c_i in enumerate(self.contexts):
                for j, c_j in enumerate(self.contexts):
                    prefix_c_ = str(q) + c_i
                    l = min(len(c_j), len(prefix_c_))
                    if c_j[:l].startswith(prefix_c_[:l]):
                        mask[q, i, j] = 1
        return mask

    def update_tr_params(self):
        self.log_context_p = logsumexp(self.log_gamma, axis=0)
        self.log_context_p -= logsumexp(self.log_context_p)
        log_a = logsumexp(self.log_ksi[:-1], axis=0) - \
                logsumexp(self.log_gamma, axis=0)
        log_a -= logsumexp(log_a, axis=0)
        self.log_a = log_a
        self._check_diff_log_p()
        self.tr_trie.recount_with_log_a(self.log_a, self.contexts,
                                        self.log_context_p)

        if self._print:
            print("p_state ",
                  np.round(np.exp(logsumexp(self.log_gamma, axis=0)), 4))
            print("c_p = {}".format(np.round(np.exp(self.log_context_p), 2)))
            print("a:\n{}".format(np.round(np.exp(self.log_a), 4)))

    def _get_log_gamma_emission(self):
        log_gamma_ = np.log(np.zeros((self.T, self.n)))
        for q in range(self.n):
            log_gamma_[:, q] = logsumexp(self.log_gamma[:, self.state_c == q],
                                         axis=1)
        log_gamma_ -= logsumexp(log_gamma_, axis=1)[:, np.newaxis]
        return log_gamma_

    def update_emission_params(self):
        self.emission.update_params(self._get_log_gamma_emission())
        self.log_b[:] = self.emission.get_log_b(self.data)

    def get_hidden_states(self):
        if self.n_contexts == 1:
            return np.zeros(self.T).astype(np.int)
        log_gamma_ = self._get_log_gamma_emission()
        return np.argmax(log_gamma_, axis=1)

    def get_n_params(self):
        n_params = 1 + self.n_contexts * (self.n - 1) + \
                   self.emission.get_n_params() + (self.n_contexts - 1)
        # contexts, transition params,
        # emission params, start distribution (pi=log_context_p)
        return n_params

    def get_aic(self):
        return 2 * (self.get_n_params() - self._log_p)

    def estimate_fdr_fndr(self, threshold=0.5):
        """
        H0 - state 0
        H1 - state != 0
        :return:
        """
        for i, c in enumerate(self.contexts):
            if not c.startswith('0'):
                break
        log_expectation_fp = np.log(0.)
        log_expectation_fn = np.log(0.)
        expectation_p1 = 0.
        expectation_p0 = 0.
        for t in range(self.T):
            log_p_0 = logsumexp(self.log_gamma[t][:i])
            log_p_1 = logsumexp(self.log_gamma[t][i:])
            log_sum = np.logaddexp(log_p_0, log_p_1)
            log_p_0, log_p_1 = log_p_0 - log_sum, log_p_1 - log_sum
            if np.exp(log_p_0) < threshold:
                expectation_p1 += 1
                log_expectation_fp = np.logaddexp(log_expectation_fp, log_p_0)
            else:
                expectation_p0 += 1
                log_expectation_fn = np.logaddexp(log_expectation_fn, log_p_1)

        return np.exp(log_expectation_fp) / expectation_p1, \
               np.exp(log_expectation_fn) / expectation_p0

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
        c_order = VLHMM._get_context_order(contexts, state_order)

        return list(np.array(contexts)[c_order]), \
               log_a[state_order, :][:, c_order]

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

    def set_params(self, contexts, log_a, emission):
        self.contexts = contexts
        self.log_a = log_a
        self.emission = emission

    def sample(self, size):
        h_states = ContextTransitionTrie.sample_(size, self.contexts,
                                                 self.log_a)
        h_states = list(map(int, h_states))
        data = np.zeros((size, self.emission.n))
        for i, state in enumerate(h_states):
            data[i] = self.emission.sample(state)
        return data

    @staticmethod
    def sample_(size, contexts, log_a, type_emission="Poisson", emission=None,
                **e_params):
        n = len(log_a)
        if emission is None:
            if type_emission == "Poisson":
                emission = PoissonEmission(n_states=n)
            else:
                emission = GaussianEmission(n_states=n)

            if len(e_params) > 0:
                emission._set_params(**e_params)
            else:
                emission.set_rand_params()

        vlhmm = VLHMM(n)
        vlhmm.set_params(contexts, log_a, emission)

        data = vlhmm.sample(size)
        return data, emission

    def create_img(self, real_contexts=None, real_log_a=None, name="",
                   language="en", extension="jpeg"):

        fname_plot = "{}plot.{}".format(name, extension)
        self.plot_log_p(language=language).savefig(fname_plot)
        fname_trie = "{}predicted_trie.{}".format(name, extension)
        ContextTransitionTrie.draw_context_trie(self.contexts, self.log_a,
                                                fname_trie)

        if real_contexts is not None:
            fname_real_trie = "{}real_trie.{}".format(name, extension)
            ContextTransitionTrie.draw_context_trie(real_contexts, real_log_a,
                                                    fname_real_trie)
