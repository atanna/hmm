import numpy as np
from joblib import Parallel, delayed
from collections import defaultdict
from scipy.misc import logsumexp
from vlhmm_.forward_backward import VLHMMWang


class OneOfManyVLHMM(VLHMMWang):
    def __init__(self, parent, data, ind):
        self.parent = parent
        self.ind = ind
        self._init(data)
        print("{}: T = {}".format(self.ind, self.T))

    def _init(self, data):
        self.n = self.parent.n
        self.T = len(data)
        self.data = data
        if data.ndim < 2:
            self.data = data[:, np.newaxis]
        self.track_log_p = defaultdict(list)
        self.emission = self.parent.emission
        self.track_e_params = self.parent.track_e_params
        self.tr_trie = self.parent.tr_trie
        self.max_log_p_diff = self.parent.max_log_p_diff
        self.update_contexts()

    def count_ksi(self):
        self.log_ksi[:] = np.log(0.)
        for t in range(self.T - 1):
            for i in range(self.n_contexts):
                for q in range(self.n):
                    self._update_ksi(t, q, i)
        self.log_ksi -= \
            logsumexp(self.log_ksi, axis=(1, 2)).reshape((self.T, 1, 1))
        return logsumexp(self.log_ksi[:-1], axis=0)

    def update_contexts(self):
        self.n_contexts = self.parent.n_contexts
        self.log_a = self.parent.log_a
        self.contexts = self.parent.contexts
        self.log_context_p = self.parent.log_context_p
        self.state_c = self.parent.state_c
        self.id_c = self.parent.id_c
        self.tr_trie = self.parent.tr_trie
        self._init_auxiliary_params()

    def _e_step(self):
        # raise NameError("e_step")
        super()._e_step()
        return self.log_gamma, self._log_p


class MultiVLHMM(VLHMMWang):

    def _prepare_to_fitting(self, arr_data, **_kwargs):
        kwargs = _kwargs.copy()
        self.T = sum(len(data) for data in arr_data)
        X = kwargs.pop("X", None)
        self.X = np.concatenate(X) if X is not None else None

        self.data = np.concatenate(arr_data)

        super()._prepare_to_fitting(self.data, X=self.X, **kwargs)

        self.n_vlhmms = len(arr_data)
        self.vlhmms = []
        for i, data in enumerate(arr_data):
            self.vlhmms.append(OneOfManyVLHMM(self, data, i))

    def _e_step(self):
        log_gamma, log_p = zip(
            *Parallel(n_jobs=1)(
                delayed(vlhmm._e_step)()
                for vlhmm in self.vlhmms))
        self.log_gamma = np.concatenate(log_gamma)
        self._log_p = np.sum(log_p)

        self.track_log_p[self.n_contexts].append(self._log_p)

        self._check_diff_log_p()

    def _m_step(self):
        arr_sum_ksi = Parallel(n_jobs=-1)(
            delayed(vlhmm.count_ksi)()
            for vlhmm in self.vlhmms)
        self.sum_ksi = logsumexp(arr_sum_ksi, axis=0)
        super()._m_step()

    def update_tr_params(self):
        log_sum_gamma = logsumexp(self.log_gamma, axis=0)
        self.log_context_p[:] = log_sum_gamma
        self.log_context_p -= logsumexp(self.log_context_p)
        log_a = self.sum_ksi - log_sum_gamma
        self.log_a[:] = log_a - logsumexp(log_a, axis=0)

        self.tr_trie.recount_with_log_a(self.log_a, self.contexts,
                                        self.log_context_p)

        print("c_p = {}".format(np.round(np.exp(self.log_context_p), 2)))
        print("a{}".format(np.round(np.exp(self.log_a), 2)))

    def _prune(self, th_prune):
        res = super()._prune(th_prune)
        # Parallel(n_jobs=1)(
        #     delayed(vlhmm.update_contexts)()
        #     for vlhmm in self.vlhmms)
        for vlhmm in self.vlhmms:
            vlhmm.update_contexts()
        return res

    def set_canonic_view(self):
        super().set_canonic_view()
        for vlhmm in self.vlhmms:
            vlhmm.emission = self.emission
            vlhmm.update_contexts()


