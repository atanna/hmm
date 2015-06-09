import numpy as np
import vlhmm_._vlhmmc as _vlhmmc
from joblib import Parallel, delayed
from scipy.misc import logsumexp
from vlhmm_.vlhmm import VLHMM
from vlhmm_.emission import PoissonEmission, GaussianEmission


class _Estep():
    def _e_step_for_one_sample(self, data, log_a,
                               state_c, mask, log_context_p,
                               emission, i=-1):
        T = len(data)
        n, n_contexts = log_a.shape
        log_alpha = np.log(np.zeros((T, n_contexts)))
        log_beta = np.log(np.zeros((T, n_contexts)))
        log_ksi = np.log(np.zeros((T, n, n_contexts)))
        log_b = emission.get_log_b(data)

        # _vlhmmc._e_step(mask, log_a, log_b, log_context_p, state_c,
        #                 log_alpha, log_beta, log_ksi)

        _vlhmmc._log_forward(mask,
                             log_a, log_b,
                             log_context_p,
                             state_c, log_alpha)
        _vlhmmc._log_backward(mask, log_a, log_b, log_beta)
        _vlhmmc._log_ksi(mask,
                         log_a, log_b, log_alpha,
                         log_beta, log_ksi)

        _log_p = logsumexp(log_alpha[-1])

        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1)[:, np.newaxis]

        log_sum_ksi = logsumexp(log_ksi[:-1], axis=0)
        del T, n, n_contexts, log_alpha, log_beta, log_ksi, log_b
        del data, log_a, state_c, emission
        return log_gamma, _log_p, log_sum_ksi


class MultiVLHMM(VLHMM):
    def _prepare_to_fitting(self, arr_data, **_kwargs):
        kwargs = _kwargs.copy()
        self.T = sum(len(data) for data in arr_data)
        X = kwargs.pop("X", None)
        self.X = np.concatenate(X) if X is not None else None

        self.arr_data = arr_data
        data = np.concatenate(arr_data)

        super()._prepare_to_fitting(data, X=self.X, **kwargs)

    def get_hidden_states(self):
        states = super().get_hidden_states()
        arr_T = [len(data) for data in self.arr_data]
        t = 0
        res = []
        for T in arr_T:
            res.append(states[t: t + T])
            t += T
        return res

    def fit(self, data, parallel_params=dict(n_jobs=-1, backend='threading'),
            **kwargs):
        self.parallel_params = parallel_params
        super().fit(data, **kwargs)
        return self

    def _e_step(self):
        self.mask = self.get_context_mask()

        estep = _Estep()
        log_gamma, log_p, log_sum_ksi = zip(
            *Parallel(**self.parallel_params)(
                delayed(estep._e_step_for_one_sample)(data,
                                                      self.log_a,
                                                      self.state_c,
                                                      self.mask,
                                                      self.log_context_p,
                                                      self.emission,
                                                      i=i)
                for i, data in enumerate(self.arr_data)))

        self.log_gamma = np.concatenate(log_gamma)
        self._log_p = np.sum(log_p)
        self.track_log_p[self.n_contexts].append(self._log_p)
        self._check_diff_log_p()
        self.sum_ksi = logsumexp(log_sum_ksi, axis=0)
        del log_gamma, log_p, log_sum_ksi

    def update_tr_params(self):
        log_sum_gamma = logsumexp(self.log_gamma, axis=0)
        self.log_context_p[:] = log_sum_gamma
        self.log_context_p -= logsumexp(self.log_context_p)
        log_a = self.sum_ksi - log_sum_gamma
        self.log_a[:] = log_a - logsumexp(log_a, axis=0)

        self.tr_trie.recount_with_log_a(self.log_a, self.contexts,
                                        self.log_context_p)

        if self._print:
            print("c_p = {}".format(np.round(np.exp(self.log_context_p), 2)))
            print("a{}".format(np.round(np.exp(self.log_a), 2)))

    def update_emission_params(self):
        self.emission.update_params(self._get_log_gamma_emission())

    @staticmethod
    def sample_(arr_sizes, contexts, log_a, type_emission="Poisson", emission=None,
                **e_params):
        n = len(log_a)
        if emission is None:
            if type_emission == "Poisson":
                emission = PoissonEmission(n_states=n)
            else:
                emission = GaussianEmission(n_states=n)

            if len(e_params) > 0:
                emission._set_params(**e_params)
            else:
                emission.set_rand_params()

        vlhmm = VLHMM(n)
        vlhmm.set_params(contexts, log_a, emission)

        arr_data = [vlhmm.sample(size) for size in arr_sizes]
        return arr_data, emission

