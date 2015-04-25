import datrie
import random
import numpy as np
from pygraphviz import AGraph
from scipy.misc import logsumexp


class ContextTransitionTrie():
    def __init__(self, data=None, n=2, max_len=3, **kwargs):
        self.n = n
        self._max_len = max_len
        self.alphabet = "".join(list(map(str, range(self.n))))
        self._init_seq_contexts()

        if data is None:
            self._init_without_data(**kwargs)
        else:
            self._init_tr_trie(data)

    def _init_seq_contexts(self):
        def gen_all_contexts(alphabet, l):
            if l == 0:
                yield ""
            else:
                for q in alphabet:
                    for c in gen_all_contexts(alphabet, l-1):
                        yield q+c
        self.seq_contexts = list(gen_all_contexts(self.alphabet, self._max_len))
        self.n_contexts = len(self.seq_contexts)

    def _init_without_data(self, **kwargs):
        start = kwargs.get("start", "equal")
        log_a = self.count_log_a(type=start)
        self.recount_with_log_a(log_a, self.seq_contexts)

    def _init_tr_trie(self, data):
        def freq(w, eps=1e-8):
            try:
                res = trie[w]
                return res if res != 0 else eps
            except KeyError:
                return eps

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

        trie = self._build_trie(data)

        self.log_a = np.log(np.zeros((self.n, self.n_contexts)))
        for i, c in enumerate(self.seq_contexts):
            for q in self.alphabet:
                self.log_a[int(q), i] = log_tr_p(q, c)
        self.log_a = self.log_a - logsumexp(self.log_a, axis=0)
        log_c_p = np.log(np.array([freq(c) for c in self.seq_contexts]))
        log_c_p -= logsumexp(log_c_p)
        self.recount_with_log_a(self.log_a, self.seq_contexts, log_c_p)

    def _build_trie(self, data):
        """
        :param data:
        :param max_len:
        the maximum length of the context (letter M in Dumont)
        :param min_num: default=2
        the minimum number of occurrences for context
        :return:
        """
        print("build trie...")
        _data = data[::-1]
        self._end = _data[:self._max_len]
        trie = datrie.Trie(self.alphabet)
        self.T = len(_data)
        min_num=1
        term_nodes = set()
        for i in range(self.T):
            for l in range(1, self._max_len + 2):
                if i + l > self.T:
                    break
                s = _data[i: i + l]
                if s in trie:
                    trie[s] += 1
                else:
                    trie[s] = 1
                if l == self._max_len:
                    term_nodes.add(_data[i: i + self._max_len])

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
                if len(v) < self._max_len:
                    for q in self.alphabet:
                        if v+q not in trie:
                            trie[v+q] = 0
                            term_nodes.add(v+q)

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
                res = np.logaddexp(res, log_p+self.contexts[w[1:]])
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
            assert denom > -np.inf, "q={} s={}  {}\n{}\n{}"\
                .format(q, s, res, self.log_c_tr_trie.items(),
                        self.contexts.items())
            res = res - denom
            self._log_tr_trie[qs] = res
            return res

    def get_c(self, w, direction=1):
        s = w[::direction]
        try:
            return self.contexts.longest_prefix(s)
        except KeyError:
            assert False, "{} - {} \n{} {}".format(s, self.contexts.items(s), self._max_len, self.contexts.keys())
            candidates = self.contexts.items(s)
            if len(candidates) == 0:
                return self.get_c(s[:-1])
            return sorted(candidates, key=lambda x: -x[1])[0][0]
            # # first least context with the same prefix

    def get_list_c(self, s):
        try:
            return [self.contexts.longest_prefix(s)]
        except KeyError:
            candidates = self.contexts.keys(s)
            return candidates

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

    def count_log_a(self, type=""):
        if type == "equal" :
            log_a = np.ones((self.n, self.n_contexts))
        elif type == "rand" or "log_c_tr_trie" not in self.__dict__:
            log_a = np.random.random((self.n, self.n_contexts))
        else:
            log_a = np.array(
                [[self.log_tr_p(q, self.seq_contexts[l])
                  for l in range(self.n_contexts)]
                 for q in self.alphabet])
        return log_a - logsumexp(log_a, axis=0)

    def recount_with_log_a(self, log_a, seq_contexts, log_c_p=None):
        """
        recount transition probability using matrix of probability
        :return:
        """
        self.log_c_tr_trie = datrie.Trie(self.alphabet)
        self.seq_contexts = seq_contexts
        self.n_contexts = len(seq_contexts)
        self._max_len = max(list(map(len, self.seq_contexts)))
        for i, c in enumerate(seq_contexts):
            for q in self.alphabet:
                self.log_c_tr_trie[q + c] = log_a[q, i]
        self.contexts = datrie.Trie(self.alphabet)
        if log_c_p is None:
            log_c_p = np.log(np.ones(self.n_contexts)/self.n_contexts)
        for i, c in enumerate(seq_contexts):
            self.contexts[c] = log_c_p[i]
        self.n_contexts = len(self.contexts)
        self.recount_tr_trie()
        return self

    def sample(self, size):
        X = []
        states = range(self.n)
        max_len = self._max_len
        long_context = ""
        context = ""
        for i in range(size):
            p = [self.log_tr_p(i, context) for i in self.alphabet]
            # print(context, np.exp(p))
            q = str(np.random.choice(states, p=np.exp(p)))
            X.append(q)
            long_context = (str(q) + long_context)[:max_len+1]
            cs = self.get_list_c(long_context)
            if len(cs) == 0:
                context = ""
            else:
                if len(long_context) >= max_len:
                    context = self.get_c(long_context)
                else:
                    cs = self.get_list_c(long_context)
                    context = cs[random.randrange(len(cs))]

        return "".join(map(str, X))

    @staticmethod
    def sample_(size, contexts, log_a, n=2):
        return ContextTransitionTrie(n=n).recount_with_log_a(log_a, contexts).sample(size)

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
                    changed_tr[q+s] = self.log_tr_p(q, s)
                    for q_ in self.alphabet:
                        c_to_del[q+s+q_]=1
                return True
            return False

        if self.n_contexts < 2:
            return False
        n_prune = 0
        used = set()
        changed_tr = datrie.Trie(self.alphabet)
        c_to_del = datrie.Trie(self.alphabet)
        for c in self.contexts.keys():
            if f(c[:-1]):
                n_prune += 1
        if n_prune == 0:
            return False
        for qc, log_p in changed_tr.items():
            if len(c_to_del.prefixes(qc)) == 0:
                self.log_c_tr_trie[qc] = log_p
        for c in c_to_del.keys():
            for s in list(self.log_c_tr_trie.keys(c)):
                self.log_c_tr_trie._delitem(s)
        contexts = datrie.Trie(self.alphabet)
        for s in self.log_c_tr_trie.keys():
            c = s[1:]
            _, log_c_p = zip(*self.contexts.items(c))
            contexts[c] = logsumexp(log_c_p)
        self.contexts = contexts
        self._upd_c()
        self.recount_tr_trie()
        return n_prune > 0

    def kl(self, s, q_):
        sum = 0
        for q in self.alphabet:
            sum += self.tr_p(q, s+q_)\
                   * (self.log_tr_p(q, s+q_)-self.log_tr_p(q, s))
        _, log_c_p = zip(*self.contexts.items(s))
        return np.exp(logsumexp(log_c_p)) * sum

    def draw(self, fname=None):
        return self.draw_context_trie(self.seq_contexts,
                                      self.count_log_a(),
                                      fname)

    @staticmethod
    def draw_context_trie(contexts, log_a, fname=None):
        G = AGraph(directed=True)
        n = len(log_a)
        a = np.exp(log_a)
        for c in contexts:
            v2 = c
            for i in range(len(c)):
                v1 = v2[:-1]
                if v1 == "":
                    v1 = "root"
                G.add_edge(v1, v2, label=v2[-1])
                v2= v1

        for i, c_ in enumerate(contexts):
            if c_ == "":
                c = "root"
                G.add_node(c)
            else:
                c = c_
            n_c = G.get_node(c)
            n_c.attr["style"] = "filled"
            for q in range(n):
                label = "p({}/ {})".format(q, c_)
                v2 = "p{}{}".format(q, c)
                G.add_edge(c, v2, label=label)
                n_v = G.get_node(v2)
                n_v.attr["shape"] = "box"
                n_v.attr["label"] = round(a[q, i], 2)
                n_v.attr["width"] = 0.2
                n_v.attr["height"] = 0.1
        if fname is not None:
            G.draw(fname, prog="dot")
        return G

