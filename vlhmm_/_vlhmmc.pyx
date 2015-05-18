import cython
import time
import numpy as np
cimport numpy as np
from libc.math cimport exp, log
from scipy.misc import logsumexp

ctypedef np.float64_t dtype_t
cdef dtype_t LOG_0 = -np.inf

@cython.boundscheck(False)
cdef inline dtype_t _max(dtype_t[:] arr) nogil:
    # find maximum value (builtin 'max' is unrolled for speed)
    cdef dtype_t value
    cdef dtype_t vmax = LOG_0
    cdef int j
    for j in range(arr.shape[0]):
        value = arr[j]
        if value > vmax:
            vmax = value
    return vmax


@cython.boundscheck(False)
cdef dtype_t _logsumexp(dtype_t[:] arr) nogil:
    cdef dtype_t vmax = _max(arr)
    cdef dtype_t power_sum = 0
    cdef int i
    for i in range(arr.shape[0]):
        power_sum += exp(arr[i] - vmax)

    return log(power_sum) + vmax


@cython.boundscheck(False)
cdef inline dtype_t _max_ksi(dtype_t[:,:] arr) nogil:
    cdef dtype_t value
    cdef dtype_t vmax = LOG_0
    cdef int i, j
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            value = arr[i,j]
            if value > vmax:
                vmax = value
    return vmax


@cython.boundscheck(False)
cdef void normalize_log_ksi(dtype_t[:,:,:] log_ksi):
    cdef dtype_t[:,:] _ksi_i
    cdef int i, j, k
    cdef dtype_t  vmax, tmp
    for i in range(log_ksi.shape[0]):
        _ksi_i = log_ksi[i]
        vmax = _max_ksi(_ksi_i)
        tmp = 0.
        for j in range(log_ksi.shape[1]):
            for k in range(log_ksi.shape[2]):
                tmp += exp(_ksi_i[j, k] - vmax)
        tmp = (log(tmp) + vmax)
        for j in range(log_ksi.shape[1]):
            for k in range(log_ksi.shape[2]):
                log_ksi[i,j,k] -= tmp


@cython.boundscheck(False)
cdef void fill(dtype_t[:] arr, dtype_t val) nogil:
    for i in range(arr.shape[0]):
        arr[i] = val



@cython.boundscheck(False)
def _log_forward(np.ndarray[np.uint8_t, ndim=3] mask,
                 np.ndarray[dtype_t, ndim=2] log_a,
                 np.ndarray[dtype_t, ndim=2] log_b,
                 np.ndarray[dtype_t, ndim=1] log_c_p,
                 np.ndarray[np.uint8_t, ndim=1] state_c,
                 np.ndarray[dtype_t, ndim=2] log_alpha):
    cdef int T = log_b.shape[0]
    cdef int n = log_a.shape[0]
    cdef int n_contexts = log_a.shape[1]
    cdef int t, i, j, q
    for i in range(n_contexts):
        t = 0
        log_alpha[t, i] = log_c_p[i] + log_b[t, state_c[i]]
    cdef dtype_t[:] tmp = np.zeros(n_contexts)
    cdef dtype_t log_transition

    with nogil:
        for t in range(T - 1):
            for i in range(n_contexts):
                q = state_c[i]
                for j in range(n_contexts):
                    tmp[j] = LOG_0
                    if mask[q, j, i]:
                        log_transition = log_a[q, j]
                        tmp[j] = log_alpha[t, j] + log_transition
                log_alpha[t+1, i] = _logsumexp(tmp) + log_b[t + 1, q]


@cython.boundscheck(False)
def _log_backward(np.ndarray[np.uint8_t, ndim=3] mask,
                  np.ndarray[dtype_t, ndim=2] log_a,
                  np.ndarray[dtype_t, ndim=2] log_b,
                  np.ndarray [dtype_t, ndim=2] log_beta):
    log_beta[-1] = 0.
    cdef int T = log_b.shape[0]
    cdef int n = log_a.shape[0]
    cdef int n_contexts = log_a.shape[1]
    cdef int t, i, j, q
    cdef dtype_t[:] tmp = np.zeros(n_contexts)
    with nogil:
        for t in range(T - 2, -1, -1):
            for i in range(n_contexts):
                fill(tmp, LOG_0)
                for q in range(n):
                    for j in range(n_contexts):
                        if mask[q, i, j]:
                            tmp[j] = log_a[q, i] + log_b[t+1, q] + log_beta[t+1, j]
                log_beta[t, i] = _logsumexp(tmp)


@cython.boundscheck(False)
def _log_ksi(np.ndarray[np.uint8_t, ndim=3] mask,
             np.ndarray[dtype_t, ndim=2] log_a,
             np.ndarray[dtype_t, ndim=2] log_b,
             np.ndarray[dtype_t, ndim=2] log_alpha,
             np.ndarray[dtype_t, ndim=2] log_beta,
             np.ndarray[dtype_t, ndim=3] log_ksi):
    cdef int T = log_b.shape[0]
    cdef int n = log_a.shape[0]
    cdef int n_contexts = log_a.shape[1]
    cdef int t, i, j, q
    cdef dtype_t[:] tmp = np.zeros(n_contexts)
    with nogil:
        for t in range(T - 1):
            for i in range(n_contexts):
                for q in range(n):
                    for j in range(n_contexts):
                        tmp[j] = LOG_0
                        if mask[q, i, j]:
                            tmp[j] = log_alpha[t, i] + log_a[q, i] + \
                                      log_b[t+1, q] + log_beta[t+1, j]
                    log_ksi[t, q, i] = _logsumexp(tmp)
    normalize_log_ksi(log_ksi)





@cython.boundscheck(False)
cdef inline dtype_t _log_sum_p(str w, contexts, log_c_tr_trie):
    cdef str s = w
    if len(log_c_tr_trie.prefixes(w)) > 0:
        s = log_c_tr_trie.longest_prefix(w)
    cdef np.ndarray[dtype_t, ndim=1] res = np.zeros(len(log_c_tr_trie))
    res.fill(np.log(0.))
    cdef int i
    cdef float log_p
    for i,(w, log_p) in enumerate(log_c_tr_trie.items(s)):
        res[i] = log_p + contexts[w[1:]]
    return _logsumexp(res)


def log_tr_p(str q,
             str s,
             contexts,
             log_c_tr_trie,
             int n_states):

        if len(contexts.prefixes(s)) > 0:
            s = contexts.longest_prefix(s)
        cdef  np.ndarray[dtype_t, ndim=1] log_p_children
        log_p_children = np.array(
            [_log_sum_p(str(q_)+s, contexts, log_c_tr_trie)
             for q_ in range(n_states)])
        cdef dtype_t denom = _logsumexp(log_p_children)
        assert denom > -np.inf, "q={} s={}\nc_tr_trie: {}\ncontexts: {}"\
            .format(q, s,  log_c_tr_trie.items(),
                    contexts.items())
        return log_p_children[int(q)] - denom


@cython.boundscheck(False)
def _log_forward_more_score_less_speed(np.ndarray[np.uint8_t, ndim=3] mask,
                 np.ndarray[dtype_t, ndim=2] log_a,
                 np.ndarray[dtype_t, ndim=2] log_b,
                 np.ndarray[dtype_t, ndim=1] log_c_p,
                 context_trie,
                 log_c_tr_trie,
                 np.ndarray[np.uint8_t, ndim=1] state_c,
                 np.ndarray[dtype_t, ndim=2] log_alpha):
    log_alpha.fill(LOG_0)
    cdef int T = log_b.shape[0]
    cdef int n = log_a.shape[0]
    cdef int n_contexts = log_a.shape[1]
    cdef int t, i, j, q
    cdef str c_
    cdef list contexts = context_trie.keys()
    for i in range(n_contexts):
        t = 0
        log_alpha[t, i] = log_c_p[i] + log_b[t, state_c[i]]
    cdef np.ndarray[dtype_t, ndim=1] tmp = np.zeros(n_contexts)
    cdef dtype_t log_transition
    for t in range(T - 1):
        for i in range(n_contexts):
            q = state_c[i]
            for j in range(n_contexts):
                tmp[j] = LOG_0
                c_ = contexts[j]
                if mask[q, j, i]:
                    log_transition = log_a[q, j]
                    if len(c_) > t:
                        log_transition = log_tr_p(str(q), c_[:t+1], context_trie,
                                                  log_c_tr_trie, n)
                    tmp[j] = log_alpha[t, j] + log_transition
            log_alpha[t+1, i] = _logsumexp(tmp) + log_b[t + 1, q]



def e_step(np.ndarray[np.uint8_t] mask,
           np.ndarray[dtype_t, ndim=2] log_a,
           np.ndarray[dtype_t, ndim=2] log_b,
           np.ndarray[dtype_t, ndim=1] log_context_p,
           np.ndarray[np.uint8_t, ndim=1] state_c):
    cdef int T = log_b.shape[0]
    cdef int n = log_a.shape[0]
    cdef int n_contexts = log_a.shape[1]
    log_alpha = np.log(np.zeros((T, n_contexts)))
    log_beta = np.log(np.zeros((T, n_contexts)))
    log_ksi = np.log(np.zeros((T, n, n_contexts)))

    _log_forward(mask, log_a, log_b, log_context_p, state_c, log_alpha)
    _log_backward(mask, log_a, log_b, log_beta)
    _log_ksi(mask, log_a, log_b, log_alpha, log_beta, log_ksi)

    _log_p = logsumexp(log_alpha[-1])
    log_gamma = log_alpha + log_beta
    log_gamma -= logsumexp(log_gamma, axis=1)[:, np.newaxis]
    log_sum_ksi = logsumexp(log_ksi[:-1], axis=0)

    return log_gamma, _log_p, log_sum_ksi

