import numpy as np
from joblib import Parallel, delayed
from scipy.misc import logsumexp
from vlhmm_.forward_backward import VLHMMWang
import vlhmm_._vlhmmc as _vlhmmc


class MultiVLHMM(VLHMMWang):
    def _prepare_to_fitting(self, arr_data, **_kwargs):
        kwargs = _kwargs.copy()
        self.T = sum(len(data) for data in arr_data)
        X = kwargs.pop("X", None)
        self.X = np.concatenate(X) if X is not None else None

        self.arr_data = arr_data
        data = np.concatenate(arr_data)

        super()._prepare_to_fitting(data, X=self.X, **kwargs)

    def _e_step_for_one_sample(self, data):
        T = len(data)
        log_alpha = np.log(np.zeros((T, self.n_contexts)))
        log_beta = np.log(np.zeros((T, self.n_contexts)))
        log_ksi = np.log(np.zeros((T, self.n, self.n_contexts)))
        log_b = self.emission.get_log_b(data)

        _vlhmmc._log_forward(self.contexts, self.log_a, log_b,
                             self.log_context_p, self.tr_trie, self.id_c,
                             self.state_c, log_alpha)

        _log_p = logsumexp(log_alpha[-1])
        _vlhmmc._log_backward(self.log_a, log_b, self.tr_trie.contexts,
                              self.id_c, self.state_c, log_beta)

        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1)[:, np.newaxis]

        _vlhmmc._log_ksi(self.log_a, log_b, self.tr_trie.contexts,
                         self.id_c, self.state_c, log_alpha,
                         log_beta, log_ksi)
        return log_gamma, _log_p, logsumexp(log_ksi[:-1], axis=0)

    def _e_step(self):
        log_gamma, log_p, log_sum_ksi = zip(
            *Parallel(n_jobs=-1)(
                delayed(self._e_step_for_one_sample)(data)
                for data in self.arr_data))
        self.log_gamma = np.concatenate(log_gamma)
        self._log_p = np.sum(log_p)
        self.track_log_p[self.n_contexts].append(self._log_p)
        self._check_diff_log_p()
        self.sum_ksi = logsumexp(log_sum_ksi, axis=0)

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

    def update_emission_params(self):
        self.emission.update_params(self._get_log_gamma_emission())





