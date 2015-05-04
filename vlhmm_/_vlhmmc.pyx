import numpy as np
cimport numpy as np
from scipy.misc import logsumexp

def _log_forward(contexts,
                 np.ndarray[np.float64_t, ndim=2] log_a,
                 np.ndarray[np.float64_t, ndim=2] log_b,
                 np.ndarray[np.float64_t, ndim=1] log_context_p,
                 tr_trie, id_c, state_c,
                 np.ndarray[np.float64_t, ndim=2] log_alpha):
        log_alpha[:] = -np.inf
        cdef int n_contexts = len(log_context_p)
        cdef int T = len(log_b)
        for i in range(n_contexts):
            t = 0
            log_alpha[t][i] = \
                log_context_p[i] + \
                log_b[t, state_c[i]]
        for t in range(T - 1):
            for i in range(n_contexts):
                log_alpha[t + 1][i] = np.log(0.)
                c = contexts[i]
                for c_ in tr_trie.get_list_c(c[1:]):
                    i_ = id_c[c_]
                    log_transition = log_a[state_c[i], i_]
                    if len(c_) > t:
                        log_transition = tr_trie.log_tr_p(c[0], c_[:t+1])
                    log_alpha[t + 1][i] = np.logaddexp(
                        log_alpha[t + 1][i],
                        log_alpha[t][i_]
                        + log_transition
                        + log_b[t + 1, state_c[i]])


def _log_backward(contexts,
                  np.ndarray[np.float64_t, ndim=2] log_a,
                  np.ndarray[np.float64_t, ndim=2] log_b,
                  tr_trie, id_c, state_c,
                  np.ndarray [np.float64_t, ndim=2] log_beta):
    log_beta[:] = np.log(0.)
    log_beta[-1] = np.log(1.)
    cdef int T = len(log_b)
    cdef int n = log_a.shape[0]
    cdef int n_contexts = log_a.shape[1]
    for t in range(T - 2, -1, -1):
        for i in range(n_contexts):
            log_beta[t][i] = np.log(0.)
            c = contexts[i]
            for q in range(n):
                for c_ in tr_trie.get_list_c(str(q) + c):
                    i_ = id_c[c_]
                    log_beta[t][i] = np.logaddexp(
                        log_beta[t][i],
                        log_a[q, i]
                        + log_b[t + 1, q]
                        + log_beta[t + 1][i_])


def _log_ksi(contexts,
             np.ndarray[np.float64_t, ndim=2] log_a,
             np.ndarray[np.float64_t, ndim=2] log_b,
             tr_trie, id_c, state_c,
             np.ndarray[np.float64_t, ndim=2] log_alpha,
             np.ndarray[np.float64_t, ndim=2] log_beta,
             np.ndarray[np.float64_t, ndim=3] log_ksi):
    log_ksi[:] = np.log(0.)
    cdef int T = len(log_b)
    cdef int n = log_a.shape[0]
    cdef int n_contexts = log_a.shape[1]
    for t in range(T - 1):
        for i in range(n_contexts):
            for q in range(n):
                log_ksi[t][q, i] = np.log(0.)
                for c_ in tr_trie.get_list_c(str(q)+contexts[i]):
                    i_ = id_c[c_]
                    log_ksi[t][q, i] = \
                        np.logaddexp(log_ksi[t][q, i],
                                     log_alpha[t][i] +
                                     log_a[q, i] +
                                     log_b[t+1, q] +
                                     log_beta[t+1, i_])
    log_ksi -=  \
        logsumexp(log_ksi, axis=(1, 2)).reshape((T, 1, 1))

