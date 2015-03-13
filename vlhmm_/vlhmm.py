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
    def __init__(self, data, labels, L=None):
        if L is None:
            self.L = len(set(labels))
        else:
            self.L = L
        self.labels = np.array(list(labels)).astype(np.int)
        self.data = data
        self._init_params(data)

    def _init_params(self, data, eps=1e-4):
        self.n = data.shape[1]
        self.mu = np.zeros((self.L, self.n))
        self.sigma = np.zeros((self.L, self.n, self.n))
        self.T = len(data)
        for l in range(self.L):
            y = data[self.labels == l]
            self.mu[l] = y.mean(axis=0)
            tmp = (y - self.mu[l]) + eps
            self.sigma[l] = np.array((tmp.T.dot(tmp)) / len(y)) \
                .reshape((self.n, self.n))

    def update_params(self, log_gamma):
        def get_new_params(l):
            mu = (gamma[:, l][:, np.newaxis] * self.data).sum(axis=0)
            denom = gamma[:, l].sum()
            mu /= denom
            sigma = np.zeros((self.n, self.n))
            for t in range(self.T):
                tmp = (self.data[t] - mu)[:, np.newaxis]
                sigma += gamma[t, l] \
                         * tmp.dot(tmp.T)
            return mu, sigma / denom

        gamma = np.exp(log_gamma)
        for l in range(self.L):
            self.mu[l], self.sigma[l] = get_new_params(l)

    def log_p(self, y, l):
        return multivariate_normal.logpdf(y, self.mu[l], self.sigma[l])


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
        self._end = _data[:max_len]
        trie = datrie.Trie(self.alphabet)
        self._n = len(_data)
        self._max_len = max_len
        term_nodes = set()
        for i in range(self._n):
            for l in range(1, max_len + 2):
                if i + l > self._n:
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
                    for c in self.alphabet:
                        if v+c not in trie:
                            trie[v+c] = 0
                            term_nodes.add(v+c)
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
        return self.N(w) / self._n

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
        return np.log(self.N(w)) - np.log(self._n)

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


class VLHMM():
    def __init__(self, n):
        """
        :param n: - length of context's alphabet, n <= 10
        :return:
        """
        self.n = n

    def _init_X(self, data):
        centre, labels = kmeans2(data, self.n)
        self.X = "".join(list(map(str, labels)))

    def get_c(self, *args):
        return self.context_trie.get_c(
            reduce(lambda res, x: res + str(x), args, ""))


len_print = 4


