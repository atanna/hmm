from collections import defaultdict
import numpy as np
from scipy.misc import logsumexp
from vlhmm_.forward_backward import VLHMMWang


class OneOfManyVLHMM(VLHMMWang):
    def _init(self, data):
        self.T = len(data)
        self.data = data
        if data.ndim < 2:
            self.data = data[:, np.newaxis]
        self.n_contexts = self._get_n_contexts()
        self._init_auxiliary_params()
        self.track_log_p = defaultdict(list)
        self.track_e_params = {}

    def _get_n_contexts(self):
        return self.__dict__.get("n_contexts", self.n**self.max_len)

    def _init_emission(self, type_emission):
        pass

    def update_tr_params(self):
        self.log_ksi[:] = np.log(0.)
        for t in range(self.T - 1):
            for i in range(self.n_contexts):
                for q in range(self.n):
                    self._update_ksi(t, q, i)
        self._log_a = logsumexp(self.log_ksi[:-1], axis=0) \
                      - logsumexp(self.log_gamma, axis=0)

    def update_emission_params(self):
        pass

    def update_contexts(self):
        self.n_contexts = self.parent.n_contexts
        self.tr_trie =self.parent.tr_trie
        self.contexts = self.parent.contexts
        self.log_context_p = self.parent.log_context_p
        self.state_c = self.parent.state_c
        self.id_c = self.parent.id_c
        self.log_a = self.parent.log_a
        self._init_auxiliary_params()

    def _log_gamma(self):
        super()._log_gamma()



class MultiVLHMM(VLHMMWang):

    def _prepare_to_fitting(self, arr_data, **kwargs):
        X = kwargs.get("X")
        self.n_vlhmms = len(arr_data)
        self.vlhmms = []
        self.T = 0
        for i, data in enumerate(arr_data):
            vlhmm = OneOfManyVLHMM(self.n)
            _X = None if X is None else X[i]
            vlhmm._prepare_to_fitting(data, X=_X, **kwargs)
            vlhmm.parent = self
            self.vlhmms.append(vlhmm)
            self.T += len(data)
        if arr_data[0].ndim > 1:
            self.data = np.vstack(arr_data)
        else:
            self.data = np.hstack(arr_data)
        print("T =", self.T, self.data.shape)
        super()._prepare_to_fitting(self.data, **kwargs)
        for vlhmm in self.vlhmms:
            vlhmm.update_contexts()
            vlhmm.track_e_params = self.track_e_params

    def _init_a(self):
        super()._init_a()
        for vlhmm in self.vlhmms:
            vlhmm.log_a = self.log_a

    def _init_emission(self, *args, **kwargs):
        super()._init_emission(*args, **kwargs)
        for vlhmm in self.vlhmms:
            vlhmm.emission = self.emission

    def _e_step(self):
        self._log_p = np.log(0.)
        for vlhmm in self.vlhmms:
            vlhmm._e_step()
            self._log_p = np.logaddexp(self._log_p, vlhmm._log_p)
        self.track_log_p[self.n_contexts].append(self._log_p)

    def _m_step(self):
        for vlhmm in self.vlhmms:
            vlhmm._m_step()
        super()._m_step()

    def update_tr_params(self):
        sum_log_gamma = np.log(np.zeros((self.n_contexts)))
        log_a = np.log(np.zeros((self.n, self.n_contexts)))
        for vlhmm in self.vlhmms:
            log_a = np.logaddexp(log_a, vlhmm._log_a)
            sum_log_gamma = np.logaddexp(sum_log_gamma,
                                         logsumexp(vlhmm.log_gamma, axis=0))
        self.log_context_p = sum_log_gamma - logsumexp(sum_log_gamma)
        print("c_p = {}".format(np.round(np.exp(self.log_context_p), 2)))

        self.log_a = log_a - logsumexp(log_a, axis=0)
        self.tr_trie.recount_with_log_a(self.log_a, self.contexts,
                                        self.log_context_p)
        print(np.round(np.exp(self.log_a),2))

    def update_emission_params(self):
        t = 0.
        log_gamma = np.log(np.zeros((self.T, self.n)))
        for vlhmm in self.vlhmms:
            T = vlhmm.T
            log_gamma[t:t+T] = vlhmm._get_log_gamma_emission()
            t += T

        self.emission.update_params(log_gamma)
        print("\n\n")

    def _prune(self, th_prune):
        res = super()._prune(th_prune)
        for vlhmm in self.vlhmms:
            vlhmm.update_contexts()
        return res


