import datrie
import numpy as np

"""
All theory and notations were taken from:
http://projecteuclid.org/download/pdf_1/euclid.aos/1018031204
"""


class Context():
    def __init__(self, data, **kwargs):
        self._build_trie(data, **kwargs)
        self.choose_K()

    def _build_trie(self, data, k=3, min_num=2):
        """
        :param data:
        :param k:
        the maximum length of the context
        :param min_num: default=2
        the minimum number of occurrences for context
        :return:
        """
        _data = data[::-1]
        self.alphabet = set(_data)
        trie = datrie.Trie(self.alphabet)
        self._n = len(_data)
        self._k = k
        term_nodes = set()
        for i in range(self._n):
            for l in range(1, k + 2):
                if i + l > self._n:
                    break
                s = _data[i: i + l]
                if s in trie:
                    trie[s] += 1
                else:
                    trie[s] = 1
                if l == k:
                    term_nodes.add(_data[i: i + k])

        for v, k in trie.items():
            if k < min_num:
                trie._delitem(v)
        self._term_nodes = set()
        for t_node in term_nodes:
            self._term_nodes.add(trie.longest_prefix(t_node))
        self._trie = trie
        self.trie = self._copy_trie(trie)

    def _copy_trie(self, trie):
        tr = datrie.Trie(self.alphabet)
        for k, v in trie.items():
            if len(k) <= self._k:
                tr[k] = v
        return tr

    def choose_K(self):
        c = 2 * len(self.alphabet) + 4
        self.K = c * np.log(self._n)

    def N(self, w):
        """
        :param w:
        :return:
        number of occurrences of the string w in the data
        """
        try:
            return self._trie[w]
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
        :param w:
        :return: p(x|w)
        """
        n_xw = self.N(x + w)
        if n_xw == 0:
            return 0.
        return n_xw / self.N(w)

    def kl(self, w, u):
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
            if tmp != 0.:
                # print(x, w, u,  self.tr_p(x, w))
                res += self.tr_p(x, wu) * \
                       np.log(tmp / self.tr_p(x, w)) * \
                       self.N(wu)
        return res

    def prune(self, K=None):
        """
        prune terminal node wu to w
        if kl(w, u) < K
        :param K:
        :return: number of changes
        """
        if K is None:
            K = self.K
        changes = 0
        for t_node in set(self._term_nodes):
            if len(t_node) > 1:
                w, u = t_node[:-1], t_node[-1]
                if self.kl(w, u) < K:
                    self._term_nodes.remove(t_node)
                    self.trie._delitem(t_node)
                    self._term_nodes.add(w)
                    changes += 1
        return changes

    def get_c(self, w, direction):
        return self.trie.longest_prefix(w[::direction])


class VLMM():
    def fit(self, data, k, K=0.7, **kwargs):
        """
        :param data:
        :param k: the maximum length of the context
        :param K: threshold for Kullback-Leibler distance
        :param kwargs:
        :return:
        """
        self.k = k
        self.context = Context(data, k=k, **kwargs)
        while self.context.prune(K) > 0:
            pass
        return self

    def get_contexts(self, X):
        n = len(X)
        for i in range(n):
            yield self.context.get_c(X[max(0, i - self.k): i + 1], -1)


