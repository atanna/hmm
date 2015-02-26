import datrie
import numpy as np
from hmm_.hmm import HMM, HMMModel

"""
All theory and notations were taken from:
http://projecteuclid.org/download/pdf_1/euclid.aos/1018031204
"""

class ContextException(Exception):
    pass

class Context():
    def __init__(self, data, **kwargs):
        self._build_trie(data, **kwargs)
        self.choose_K()

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
        else:
            for v, n_v in trie.items():
                for c in self.alphabet:
                    if c+v not in trie or trie[c+v] >= min_num:
                        break
                else:
                    for c in self.alphabet:
                        trie._delitem(c+v)


        self._term_trie = datrie.Trie(self.alphabet)
        for t_node in term_nodes:
            key = trie.longest_prefix(t_node)
            self._term_trie[key] = trie[key]
        self._recount_contexts()
        self.trie = trie

    def choose_K(self):
        c = 2 * len(self.alphabet) + 4
        self.K = c * np.log(self._n)

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
        n = self.N(w)
        if n == 0:
            return self.log_p(x)

        n_xw = self.N(x + w)
        if n_xw == 0:
            return -np.inf
        if self._end.startswith(w):
            n -= 1
        return np.log(n_xw) - np.log(n)

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
        for t_node in self._term_trie.keys():
            if len(t_node) > 1:
                w, u = t_node[:-1], t_node[-1]
                if self.kl(w, u) < K:
                    self._term_trie[w] = self._term_trie[t_node]
                    self._term_trie._delitem(t_node)
                    changes += 1
        self._recount_contexts()
        return changes

    def get_c(self, w, direction=1):
        s = w[::direction]
        try:
            return self._term_trie.longest_prefix(s)
        except KeyError:
            candidates =  self._term_trie.items(s)
            if len(candidates) == 0:
                return self.get_c(s[:-1])
            return sorted(candidates, key=lambda x: -x[1])[0][0]
            # first least context with the same prefix

    def get_contexts(self, X):
        n = len(X)
        for i in range(n):
            yield self.get_c(X[max(0, i - self._max_len): i + 1], direction=-1)


class VLMM():
    def __init__(self, n=None):
        self.n = n

    def fit(self, data, max_len, K=0.7, type_vlmm="hierarchy", min_num=2, **kwargs):
        """
        :param data:
        :param max_len: the maximum length of the context
        :param K: threshold for Kullback-Leibler distance
        :param kwargs:
        :return:
        """
        self.max_len = max_len
        self._context = Context(data, max_len=max_len, min_num=min_num)
        while self._context.prune(K) > 0:
            pass
        self.c = self._context._term_trie.keys()
        self.n_contexts = len(self.c)
        self._d_contexts = dict(zip(self.c, range(self.n_contexts)))
        self._d_contexts_invert = dict(zip(self._d_contexts.values(),
                                           self._d_contexts.keys()))
        if self.n is None:
            self.n = self.n_contexts

        if type_vlmm == "hierarchy":
            self._fit_hierarchy(data, **kwargs)
        else:
            self._fit_hmm_contexts(data, **kwargs)
        return self

    def get_contexts(self, X):
        return list(self._context.get_contexts(X))

    def _fit_hierarchy(self, data, **kwargs):
        """
        fit hmm model with observations = contexts
        :param data:
        :param kwargs: params for hmm
        :return:
        """

        def get_data_hmm(data):
            return self.dict_arr(self._d_contexts, self.get_contexts(data))

        def get_str(data_hmm):
            return "".join(map(lambda x: self._d_contexts_invert[x][0],
                               data_hmm))

        self._get_data_hmm = get_data_hmm
        self._get_str = get_str
        _, model, _ = HMM(self.n).optimal_model(self._get_data_hmm(data),
                                                **kwargs)
        self._model = model

    def _fit_hmm_contexts(self, data, **kwargs):
        """
        fit hmm_model with h_states = contexts
        :param data:
        :param kwargs: params for hmm
        :return:
        """

        def get_model():
            a = np.zeros((n, n))
            b = np.zeros((n, m))
            for c1 in self.c:
                ind_c1 = self._d_contexts[c1]
                sum_p = 0
                for c2 in self.c:
                    if (c1.startswith(c2[1:])):
                        p = self._context.p(c2)
                        a[ind_c1, self._d_contexts[c2]] += p
                        sum_p += p
                b[ind_c1, d_alphabet[c1[0]]] = 1.
                a[ind_c1] /= sum_p

            return HMMModel.get_model_from_real_prob(a, b)

        alphabet = self._context.alphabet
        n = self.n_contexts
        m = len(alphabet)
        d_alphabet = dict(zip(alphabet, range(m)))
        d_alphabet_invert = dict(zip(d_alphabet.values(),
                                     d_alphabet.keys()))

        self._get_data_hmm = lambda data: self.dict_arr(d_alphabet, data)
        self._get_str = lambda data_hmm: "".join(
            self.dict_arr(d_alphabet_invert, data_hmm))

        model = get_model()
        data_hmm = self._get_data_hmm(data)
        _, model, _ = HMM(n).optimal_model(data_hmm, start_model=model,
                                           **kwargs)
        self.n = n
        self._model = model

    def sample(self, size):
        sample, _ = self._model.sample(size)
        return self._get_str(sample)

    def score(self, data):
        data_hmm = self._get_data_hmm(data)
        return HMM(self.n).observation_log_probability(self._model, data_hmm)

    @staticmethod
    def dict_arr(d, arr):
        return list(map(lambda x: d[x], arr))





