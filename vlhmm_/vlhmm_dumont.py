import datrie
import numpy as np
from vlhmm_.vlhmm import ContextTrie, VLHMM, GaussianEmission


class ContextTrieDumont(ContextTrie):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

        for s in self.contexts:
            for x in self.alphabet:
                self.log_c_tr_trie[x+s] = super().log_tr_p(x, s)

    def log_tr_p(self, x, s):
        """
        :param x:
        :param w = q_t,q_(t-1),..q_(t-|s|+1):
        :return: log_p(x|s)
        """
        try:
            s = self._term_trie.longest_prefix(s)
        except KeyError:
            s = s
        if x+s in self.log_tr_trie:
            return self.log_tr_trie[x+s]
        else:
            return np.log(0.)

    def recount_tr_trie(self):
        """
        recount transition probability on all substrings of contexts
        :return:
        """
        def log_sum_p(s):
            res = np.log(0.)
            for w, log_p in self.log_c_tr_trie.items(s):
                if log_p != -np.inf:
                    res = np.logaddexp(res, log_p)
            return res

        diff = 0.
        for s, val in self.log_tr_trie.items():
            denom = np.log(0.)
            s1 = s[1:]
            for x in self.alphabet:
                denom = np.logaddexp(denom, log_sum_p(x+s1))
            new_val = log_sum_p(s) - denom if denom > -np.inf else log_sum_p(s)
            diff = max(diff, np.abs(new_val-val))
            self.log_tr_trie[s] = log_sum_p(s) - denom if denom > -np.inf else log_sum_p(s)
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


class VLHMMDumont(VLHMM):
    def fit(self, data, **kwargs):
        self.data = data
        self.T = len(data)
        self._init_X(data)
        self.context_trie = ContextTrieDumont(self.X, **kwargs)
        self.emission = GaussianEmission(data, list(self.X))
        self._em()
        self._prune()

    def _em(self, threshold=1e-8, n_iter=10):
        for iter in range(n_iter):
            diff = self.context_trie.recount_tr_trie()
            self.context_trie.recount_c_tr_trie()
            print(iter, diff)
            if diff < threshold:
                break

    def _prune(self):
        #prunning algorithm by Wang (just for now)
        changes = self.context_trie.prune()
        while changes > 0:
            print("prune", changes)
            changes = self.context_trie.prune()


