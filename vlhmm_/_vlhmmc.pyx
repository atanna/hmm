import cython
import numpy as np
cimport numpy as np
from scipy.misc import logsumexp


def get_list_c(contexts, s):
    try:
        return [contexts.longest_prefix(s)]
    except KeyError:
        candidates = contexts.keys(s)
        return candidates

@cython.boundscheck(False)
def _log_forward(list contexts,
                 np.ndarray[np.float64_t, ndim=2] log_a,
                 np.ndarray[np.float64_t, ndim=2] log_b,
                 np.ndarray[np.float64_t, ndim=1] log_context_p,
                 tr_trie,
                 dict id_c,
                 np.ndarray[np.uint8_t, ndim=1] state_c,
                 np.ndarray[np.float64_t, ndim=2] log_alpha):
    log_alpha[:] = -np.inf
    cdef int n_contexts = len(log_context_p)
    cdef int T = len(log_b)
    cdef int i, i_, t
    cdef str c, c_
    for i in range(n_contexts):
        t = 0
        log_alpha[t][i] = \
            log_context_p[i] + \
            log_b[t, state_c[i]]
    tmp = np.zeros(log_a.shape[1])
    cdef float log_transition
    for t in range(T - 1):
        for i, c in enumerate(contexts):
            log_alpha[t+1, i] = np.log(0.)
            tmp.fill(np.log(0.))
            for c_ in tr_trie.get_list_c(c[1:]):
                i_ = id_c[c_]
                log_transition = log_a[state_c[i], i_]
                if len(c_) > t:
                    log_transition = tr_trie.log_tr_p(c[0], c_[:t+1])
                tmp[i_] = log_alpha[t, i_] + log_transition
            log_alpha[t+1, i] = logsumexp(tmp) + log_b[t + 1, state_c[i]]

@cython.boundscheck(False)
def _log_backward(np.ndarray[np.float64_t, ndim=2] log_a,
                  np.ndarray[np.float64_t, ndim=2] log_b,
                  context_trie,
                  dict id_c,
                  np.ndarray[np.uint8_t, ndim=1] state_c,
                  np.ndarray [np.float64_t, ndim=2] log_beta):
    log_beta.fill(np.log(0.))
    log_beta[-1] = np.log(1.)
    cdef int T = len(log_b)
    cdef int n = log_a.shape[0]
    cdef int n_contexts = log_a.shape[1]
    cdef int t, i, i_, q
    cdef str c, c_
    tmp = np.zeros(log_a.shape[1])
    for t in range(T - 2, -1, -1):
        for i, c in enumerate(context_trie.keys()):
            log_beta[t][i] = np.log(0.)
            tmp.fill(np.log(0.))
            for q in range(n):
                for c_ in get_list_c(context_trie, str(q) + c):
                    i_ = id_c[c_]
                    tmp[i_] = log_a[q, i] + log_b[t+1, q] + log_beta[t+1, i_]
            log_beta[t, i] = logsumexp(tmp)

@cython.boundscheck(False)
def _log_ksi(np.ndarray[np.float64_t, ndim=2] log_a,
             np.ndarray[np.float64_t, ndim=2] log_b,
             context_trie,
             dict id_c,
             np.ndarray[np.uint8_t, ndim=1] state_c,
             np.ndarray[np.float64_t, ndim=2] log_alpha,
             np.ndarray[np.float64_t, ndim=2] log_beta,
             np.ndarray[np.float64_t, ndim=3] log_ksi):
    log_ksi.fill(np.log(0.))
    cdef int T = len(log_b)
    cdef int n = log_a.shape[0]
    cdef int n_contexts = log_a.shape[1]
    cdef int t, i, q, i_
    cdef str c, c_
    tmp = np.zeros(log_a.shape[1])
    for t in range(T - 1):
        for i, c in enumerate(context_trie.keys()):
            for q in range(n):
                tmp.fill(np.log(0.))
                for c_ in get_list_c(context_trie, str(q)+c):
                    i_ = id_c[c_]
                    tmp[i_] = log_alpha[t, i] + log_a[q, i] + \
                              log_b[t+1, q] + log_beta[t+1, i_]
                log_ksi[t, q, i] = logsumexp(tmp)
    log_ksi -=  \
        logsumexp(log_ksi, axis=(1, 2)).reshape((T, 1, 1))

