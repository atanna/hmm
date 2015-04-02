import datrie
from scipy import stats
import numpy as np
from scipy.misc import logsumexp


class ContextTransitionTrie():
    def __init__(self, data=None, **kwargs):
        if data is None:
            self._init_without_data(**kwargs)
        else:
            self._init_tr_trie(data, **kwargs)

    def _init_without_data(self, **kwargs):
        def gen_all_contexts(alphabet, l):
            if l == 0:
                yield ""
            else:
                for q in alphabet:
                    for c in gen_all_contexts(alphabet, l-1):
                        yield q+c
        self._max_len = kwargs.get("max_len", 3)
        self.n = kwargs["n"]
        self.alphabet = "".join(list(map(str, range(self.n))))
        self.seq_contexts = list(gen_all_contexts(self.alphabet, self._max_len))
        self.n_contexts = len(self.seq_contexts)
        self.contexts = datrie.Trie(self.alphabet)
        for c in self.seq_contexts:
            self.contexts[c] = 1

    def _init_tr_trie(self, *args, **kwargs):
        def freq(w):
            try:
                return trie[w]
            except KeyError:
                return 0

        def log_tr_p(q, c):
            freq_c = freq(c)
            if freq_c == 0:
                return np.log(freq(q)) - np.log(self.T)

            freq_qc = freq(q + c)
            if freq_qc == 0:
                return np.log(0)
            if self._end.startswith(c):
                freq_c -= 1
            return np.log(freq_qc) - np.log(freq_c)

        trie = self._build_trie(*args, **kwargs)

        self.log_c_tr_trie = datrie.Trie(self.alphabet)
        for c in self.seq_contexts:
            for q in self.alphabet:
                self.log_c_tr_trie[q + c] = log_tr_p(q, c)

        self.recount_tr_trie()

    def _build_trie(self, data, max_len=3, min_num=2):
        """
        :param data:
        :param max_len:
        the maximum length of the context (letter M in Dumont)
        :param min_num: default=2
        the minimum number of occurrences for context
        :return:
        """
        _data = data[::-1]
        self.alphabet = sorted(list(set(_data)))
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

        for v, n_v in trie.items():
            prune = True
            for q in self.alphabet:
                if v+q not in trie or trie[v+q] >= min_num:
                    prune = False
                    break
            if prune:
                for q in self.alphabet:
                    trie._delitem(v+q)
            else:
                if len(v) < max_len:
                    for q in self.alphabet:
                        if v+q not in trie:
                            trie[v+q] = 0
                            term_nodes.add(v+q)

        self.contexts = datrie.Trie(self.alphabet)
        for t_node in term_nodes:
            key = trie.longest_prefix(t_node)
            self.contexts[key] = trie[key]
        self._upd_c()
        return trie

    def _upd_c(self):
        self.n_contexts = len(self.contexts)
        self.seq_contexts = list(self.contexts.keys())

    def tr_p(self, q, s):
        """
        :param q:
        :param s
        :return: p(q|s)
        """
        return np.exp(self.log_tr_p(q, s))

    def _log_sum_p(self, w):
        try:
            s = self.log_c_tr_trie.longest_prefix(w)
        except:
            s = w
        res = np.log(0.)
        for w, log_p in self.log_c_tr_trie.items(s):
            if log_p != -np.inf:
                res = np.logaddexp(res, log_p)
        return res

    def log_tr_p(self, q, s):
        """
        :param q:
        :param s:
        :return: log_p(q|s)
        """
        try:
            qs = self.log_c_tr_trie.longest_prefix(q + s)
        except KeyError:
            qs = q + s
        if qs in self._log_tr_trie:
            return self._log_tr_trie[qs]
        else:
            res = self._log_sum_p(q+s)
            denom = np.log(0.)
            for q in self.alphabet:
                log_sum_p_ = self._log_sum_p(q+s)
                if np.isnan(log_sum_p_):
                    continue
                denom = np.logaddexp(denom, log_sum_p_)
            assert denom > -np.inf, "{}".format(q+s)
            res = res - denom
            self._log_tr_trie[qs] = res
            return res

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

    def get_contexts(self, X):
        n = len(X)
        c = ""
        for i in range(n):
            c = self.get_c(X[i]+c, direction=-1)
            yield c

    def recount_tr_trie(self):
        """
        recount transition probability on all substrings of contexts
        :return:
        """
        self._log_tr_trie = datrie.Trie(self.alphabet)

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

    def count_log_a(self, equal=False):
        if equal or "log_c_tr_trie" not in self.__dict__:
            log_a = np.ones((self.n, self.n_contexts))
        else:
            log_a = np.array(
                [[self.log_tr_p(q, self.seq_contexts[l])
                  for l in range(self.n_contexts)]
                 for q in self.alphabet])
        return log_a - logsumexp(log_a, axis=0)

    def recount_with_log_a(self, log_a, seq_contexts):
        """
        recount transition probability using matrix of probability
        :return:
        """
        self.log_c_tr_trie = datrie.Trie(self.alphabet)
        self.seq_contexts = seq_contexts
        for i, c in enumerate(seq_contexts):
            for q in self.alphabet:
                self.log_c_tr_trie[q + c] = log_a[q, i]
        self.contexts = datrie.Trie(self.alphabet)
        for c in seq_contexts:
            self.contexts[c] = 1
        self.n_contexts = len(self.contexts)
        self.recount_tr_trie()
        return self

    def sample(self, size):
        X = []
        states = range(self.n)
        context = ""
        for i in range(size):
            p = [self.log_tr_p(i, context) for i in self.alphabet]
            # print(context, np.exp(p))
            q = stats.rv_discrete(name='custm',
                                  values=(states,
                                          np.exp(p))).rvs()
            X.append(q)
            context = self.get_c(str(q)+context)

        return "".join(map(str, X))

    @staticmethod
    def sample_(size, contexts, log_a):
        return ContextTransitionTrie(n=2).recount_with_log_a(log_a, contexts).sample(size)

    def prune(self, K=1e-4):
        def f(s):
            if s in used:
                return False
            used.add(s)
            for q_ in self.alphabet:
                if self.kl(s, q_) > K:
                    break
            else:
                for q in self.alphabet:
                    self.log_c_tr_trie[q+s] = self.log_tr_p(q, s)
                    for q_ in self.alphabet:
                        c_to_del.update(self.log_c_tr_trie.keys(q+s+q_))
                return True
            return False

        if self.n_contexts < 2:
            return False
        n_prune = 0
        used = set()
        c_to_del = set()
        for c in self.contexts.keys():
            if f(c[:-1]):
                n_prune += 1

        for c in c_to_del:
            self.log_c_tr_trie._delitem(c)
        self.contexts = datrie.Trie(self.alphabet)
        for s in self.log_c_tr_trie.keys():
            self.contexts[s[1:]] = 1
        self._upd_c()
        self.recount_tr_trie()
        return n_prune > 0

    def kl(self, s, q_):
        sum = 0
        for q in self.alphabet:
            sum += np.exp(self.log_tr_p(q, s+q_))*(self.log_tr_p(q, s+q_) - self.log_tr_p(q, s))
        return sum

