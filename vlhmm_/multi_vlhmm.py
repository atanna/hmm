from collections import defaultdict
import numpy as np
from scipy.misc import logsumexp
from vlhmm_.forward_backward import VLHMMWang


class OneOfManyVLHMM(VLHMMWang):
    def __init__(self, parent, data, ind):
        self.parent = parent
        self.ind = ind
        self._init(data)
        print("T = {}".format(self.T))

    def _init(self, data):
        self.n = self.parent.n
        self.T = len(data)
        self.data = data
        if data.ndim < 2:
            self.data = data[:, np.newaxis]
        self.track_log_p = defaultdict(list)
        self.emission = self.parent.emission
        self.track_e_params = self.parent.track_e_params
        self.tr_trie =self.parent.tr_trie
        self.max_log_p_diff = self.parent.max_log_p_diff
        self.update_contexts()

    def _e_step(self):
        super()._e_step()
        print("{}: ".format(self.ind), end="")
        self._check_diff_log_p()

    def count_ksi(self):
        self.log_ksi[:] = np.log(0.)
        for t in range(self.T - 1):
            for i in range(self.n_contexts):
                for q in range(self.n):
                    self._update_ksi(t, q, i)
        self.log_ksi -=  \
            logsumexp(self.log_ksi, axis=(1, 2)).reshape((self.T, 1, 1))
        self._sum_ksi = logsumexp(self.log_ksi[:-1], axis=0)

    def update_contexts(self):
        self.n_contexts = self.parent.n_contexts
        self.log_a = self.parent.log_a
        self.contexts = self.parent.contexts
        self.log_context_p = self.parent.log_context_p
        self.state_c = self.parent.state_c
        self.id_c = self.parent.id_c
        self._init_auxiliary_params()


class MultiVLHMM(VLHMMWang):

    def _prepare_to_fitting(self, arr_data, **_kwargs):
        kwargs = _kwargs.copy()
        self.T = sum(len(data) for data in arr_data)
        X = kwargs.pop("X", None)
        self.X = np.hstack(X) if X is not None else None

        if arr_data[0].ndim > 1:
            self.data = np.vstack(arr_data)
        else:
            self.data = np.hstack(arr_data)

        super()._prepare_to_fitting(self.data, X=self.X, **(kwargs))

        self.n_vlhmms = len(arr_data)
        self.vlhmms = []
        for i, data in enumerate(arr_data):
            self.vlhmms.append(OneOfManyVLHMM(self, data, i))

    def _e_step(self):
        self._log_p = np.log(0.)
        t = 0
        for vlhmm in self.vlhmms:
            T = vlhmm.T
            vlhmm._e_step()
            self.log_gamma[t: t+T] = vlhmm.log_gamma
            self._log_p = np.logaddexp(self._log_p, vlhmm._log_p)
            t += T
        self.track_log_p[self.n_contexts].append(self._log_p)

        self._check_diff_log_p()

    def _m_step(self):
        for vlhmm in self.vlhmms:
            vlhmm.count_ksi()
        super()._m_step()

    def update_tr_params(self):
        self.log_context_p[:] = logsumexp(self.log_gamma, axis=0)
        self.log_context_p -= logsumexp(self.log_context_p)

        log_a = np.log(np.zeros((self.n, self.n_contexts)))
        for vlhmm in self.vlhmms:
            log_a = np.logaddexp(log_a, vlhmm._sum_ksi)
        log_a -= logsumexp(self.log_gamma, axis=0)
        self.log_a[:] = log_a - logsumexp(log_a, axis=0)

        self.tr_trie.recount_with_log_a(self.log_a, self.contexts,
                                        self.log_context_p)

        print("c_p = {}".format(np.round(np.exp(self.log_context_p), 2)))
        print("log_a{}".format(np.round(np.exp(self.log_a),2)))

    def _prune(self, th_prune):
        res = super()._prune(th_prune)
        for vlhmm in self.vlhmms:
            vlhmm.update_contexts()
        return res


