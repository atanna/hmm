import datrie
import numpy as np
from functools import reduce
from scipy.cluster.vq import kmeans2
from scipy.misc import logsumexp
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


class ContextException(Exception):
    pass


class ContextTrie():
    def __init__(self, data, **kwargs):
        self._build_trie(data, **kwargs)

    def _build_trie(self, data, max_len=3, min_num=2, big_prune=False):
        """
        :param data:
        :param max_len:
        the maximum length of the context (letter M in Dumont)
        :param min_num: default=2
        the minimum number of occurrences for context
        :return:
        """
        _data = data[::-1]
        self.alphabet = set(_data)
        self.n = len(self.alphabet)
        self._end = _data[:max_len]
        trie = datrie.Trie(self.alphabet)
        self.T = len(_data)
        self._max_len = max_len
        term_nodes = set()
        for i in range(self.T):
            for l in range(1, max_len + 2):
                if i + l > self.T:
                    break
                s = _data[i: i + l]
                if s in trie:
                    trie[s] += 1
                else:
                    trie[s] = 1
                if l == max_len:
                    term_nodes.add(_data[i: i + max_len])
        if big_prune:
            for v, n_v in trie.items():
                if n_v < min_num:
                    trie._delitem(v)
            for v, n_v in trie.items():
                if len(v) < max_len:
                    not_full = False
                    for c in self.alphabet:
                        if v+c not in trie:
                            not_full=True
                    if not_full:
                        term_nodes.add(v)
        else:
            for v, n_v in trie.items():
                prune = True
                for c in self.alphabet:
                    if v+c not in trie or trie[v+c] >= min_num:
                        prune = False
                        break
                if prune:
                    for c in self.alphabet:
                        trie._delitem(v+c)
                else:
                    if len(v) < max_len:
                        for c in self.alphabet:
                            if v+c not in trie:
                                trie[v+c] = 0
                                term_nodes.add(v+c)

        self._term_trie = datrie.Trie(self.alphabet)
        for t_node in term_nodes:
            key = trie.longest_prefix(t_node)
            self._term_trie[key] = trie[key]
        self._recount_contexts()
        self.trie = trie

    def _recount_contexts(self):
        self.contexts = self._term_trie.keys()
        self.n_contexts = len(self.contexts)

    def N(self, w):
        """
        :param w:
        :return:
        number of occurrences of the string w in the data
        """
        try:
            return self.trie[w]
        except KeyError:
            return 0

    def p(self, w):
        """
        :param w:
        :return: p(w)
        """
        return self.N(w) / self.T

    def tr_p(self, x, w):
        """
        :param x:
        :param w = q_t,q_(t-1),..q_(t-|w|+1):
        :return: p(x|w)
        """
        return np.exp(self.log_tr_p(x, w))

    def log_p(self, w):
        """
        :param w:
        :return: log_p(w)
        """
        return np.log(self.N(w)) - np.log(self.T)

    def log_tr_p(self, x, w):
        """
        :param x:
        :param w = q_t,q_(t-1),..q_(t-|w|+1):
        :return: log_p(x|w)
        """
        try:
            s = self._term_trie.longest_prefix(w)
        except KeyError:
            s = w

        n = self.N(s)
        if n == 0:
            return self.log_p(x)

        n_xw = self.N(x + s)
        if n_xw == 0:
            return np.log(0)
        if self._end.startswith(s):
            n -= 1
        return np.log(n_xw) - np.log(n)

    def get_c(self, w, direction=1):
        s = w[::direction]
        try:
            return self._term_trie.longest_prefix(s)
        except KeyError:
            candidates = self._term_trie.items(s)
            if len(candidates) == 0:
                return self.get_c(s[:-1])
            return sorted(candidates, key=lambda x: -x[1])[0][0]
            # first least context with the same prefix

    def get_contexts(self, X):
        n = len(X)
        for i in range(n):
            yield self.get_c(X[max(0, i - self._max_len): i + 1], direction=-1)

    def kl(self, w, u, eps=1e-16):
        """
        Kullback-Leibler distance between two probability measures
        P(.|wu) and P(.|w)
        :param w:
        :param u:
        :return:
        """
        wu = w + u
        res = 0.
        for x in self.alphabet:
            tmp = self.tr_p(x, wu)
            if tmp > eps:
                res += self.tr_p(x, wu) * \
                       np.log(tmp / self.tr_p(x, w))
        return res * self.p(wu)

    def prune(self, K=1e-4):
        """
        prune terminal node wu to w
        if kl(w, u) < K
        :param K:
        :return: number of changes
        """
        changes = 0
        for t_node in self._term_trie.keys():
            if len(t_node) > 1:
                w, u = t_node[:-1], t_node[-1]
                if self.kl(w, u) < K:
                    self._term_trie[w] = self._term_trie[t_node]
                    self._term_trie._delitem(t_node)
                    changes += 1
        self._recount_contexts()
        return changes


class ContextTrieVLMM(ContextTrie):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contexts = self._term_trie
        self.alphabet = sorted(self.alphabet)
        self._init_tr_trie()

    def _init_tr_trie(self):
        """
        log_tr_trie = trie with log transition probability for all s,
        such that s prefix context
        log_tr_trie[s] = log_p(s[0]| s[1:])

        log_c_tr_trie = trie with log transition probability for contexts
        log_c_tr_trie[s] = log_p(s[0]| s[1:])
        :return:
        """
        self.log_tr_trie = datrie.Trie(self.alphabet)
        for s in self.trie.keys():
                self.log_tr_trie[s] = super().log_tr_p(s[0], s[1:])

        self.log_c_tr_trie = datrie.Trie(self.alphabet)

        for s in self.contexts.keys():
            for x in self.alphabet:
                self.log_c_tr_trie[x+s] = super().log_tr_p(x, s)

    def log_tr_p(self, x, s):
        """
        :param x:
        :param s = q_t,q_(t-1),..q_(t-|s|+1):
        :return: log_p(x|s)
        """
        try:
            xs = self.log_c_tr_trie.longest_prefix(x + s)
        except KeyError:
            xs = x + s
        if xs in self.log_tr_trie:
            return self.log_tr_trie[xs]
        else:
            return np.log(0.)

    def recount_tr_trie(self):
        """
        recount transition probability on all substrings of contexts
        :return:
        """

        def log_sum_p(items):
            if len(items) == 0:
                return None
            res = np.log(0.)
            for w, log_p in items:
                if log_p != -np.inf:
                    res = np.logaddexp(res, log_p)
            return res

        diff = 0.
        log_tr_trie = datrie.Trie(self.alphabet)
        for s, val in self.log_tr_trie.items():
            numer = log_sum_p(self.log_c_tr_trie.items(s))
            if numer is None:
                continue
            denom = np.log(0.)
            s1 = s[1:]
            for x in self.alphabet:
                log_sum_p_ = log_sum_p(self.log_c_tr_trie.items(x+s1))
                assert log_sum_p_ is not None, x+s1
                denom = np.logaddexp(denom, log_sum_p_)
            new_val = numer - denom if denom > -np.inf else numer
            log_tr_trie[s] = new_val
            diff = max(diff, np.abs(new_val-val))
        self.log_tr_trie = log_tr_trie
        return diff

    def recount_c_tr_trie(self):
        """
        recount transition probability on contexts
        :return:
        """
        diff = 0.
        for s, val in self.log_c_tr_trie.items():
            new_val = self.log_tr_p(s[0], s[1:])
            diff = max(diff, np.abs(new_val-val))
            self.log_c_tr_trie[s] = new_val

        return diff

    def count_log_a(self, uniform=False):
        if uniform:
            log_a = np.ones((self.n, self.n_contexts))
        else:
            contexts = sorted(self.contexts.keys())
            log_a = np.array(
                [[self.log_tr_p(q, contexts[l])
                  for l in range(self.n_contexts)]
                 for q in self.alphabet])
        return log_a - logsumexp(log_a, axis=0)

    def recount_with_log_a(self, log_a, contexts):
        """
        recount transition probability using matrix of probability
        :return:
        """
        self.log_c_tr_trie = datrie.Trie(self.alphabet)
        for i, c in enumerate(contexts):
            for q in self.alphabet:
                self.log_c_tr_trie[q + c] = log_a[q, i]
        self.contexts = datrie.Trie(self.alphabet)
        for c in contexts:
            self.contexts[c]=1
        self.n_contexts = len(self.contexts)
        self.recount_tr_trie()

    def prune(self, K=1e-4):
        def f(s):
            if s in used:
                return
            used.add(s)
            for q_ in self.alphabet:
                if self.kl(s, q_) > K:
                    break
            else:
                for q in self.alphabet:
                    self.log_c_tr_trie[q+s] = self.log_tr_trie[q+s]
                    for q_ in self.alphabet:
                        _del(q+s+q_)
                return True
            return False

        def _del(s):
            self.log_c_tr_trie._delitem(s)
            for q_ in self.alphabet:
                if s+q_ in self.log_tr_trie:
                    _del(s+q_)
        prune = False
        used = set()
        for c in self.contexts.keys():
            if f(c[:-1]):
                prune = True
        self.contexts = datrie.Trie(self.alphabet)
        for s in self.log_c_tr_trie.keys():
            self.contexts[s[1:]] = 1
        self.n_contexts = len(self.contexts)
        self.recount_tr_trie()
        return prune

    def kl(self, s, q_):
        sum = 0
        for q in self.alphabet:
            sum += np.exp(self.log_tr_trie[q+s+q_])*(self.log_tr_trie[q+s+q_] - self.log_tr_trie[q+s])
        return sum

    def get_c(self, w, direction=1):
        s = w[::direction]
        try:
            return self.contexts.longest_prefix(s)
        except KeyError:
            candidates = self.contexts.items(s)
            if len(candidates) == 0:
                return self.get_c(s[:-1])
            return sorted(candidates, key=lambda x: -x[1])[0][0]
            # first least context with the same prefix



class VLHMM():
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


len_print = 4


