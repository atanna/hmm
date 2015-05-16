import cython
import numpy as np
cimport numpy as np
from libc.math cimport exp, log
from scipy.misc import logsumexp
import time

ctypedef np.float64_t dtype_t
cdef dtype_t _NINF = -np.inf

@cython.boundscheck(False)
cdef inline dtype_t _max(np.ndarray values):
    # find maximum value (builtin 'max' is unrolled for speed)
    cdef dtype_t value
    cdef dtype_t vmax = _NINF
    for value in values:
        if value > vmax:
            vmax = value
    return vmax


@cython.boundscheck(False)
cdef dtype_t _logsumexp(np.ndarray[dtype_t, ndim=1] X):
    cdef dtype_t vmax = _max(X)
    cdef dtype_t power_sum = 0

    for i in range(X.shape[0]):
        power_sum += exp(X[i] - vmax)

    return log(power_sum) + vmax


@cython.boundscheck(False)
cdef inline dtype_t _max_ksi(np.ndarray[dtype_t, ndim=2] arr):
    cdef np.ndarray[dtype_t] row
    cdef dtype_t value
    cdef dtype_t vmax = _NINF
    for row in arr:
        for value in row:
            if value > vmax:
                vmax = value
    return vmax

@cython.boundscheck(False)
cdef np.ndarray[dtype_t] _logsumexp_ksi_12(np.ndarray[dtype_t, ndim=3] log_ksi):
    cdef np.ndarray[dtype_t] res = np.zeros(log_ksi.shape[0])
    cdef int i
    for i, _ksi in enumerate(log_ksi):
        vmax = _max_ksi(_ksi)
        tmp = 0.
        for row in _ksi:
            for value in row:
                tmp += np.exp(value - vmax)
        res[i] = log(tmp) + vmax
    return res


@cython.boundscheck(False)
cdef inline list get_list_c(contexts, s):
    try:
        return [contexts.longest_prefix(s)]
    except KeyError:
        candidates = contexts.keys(s)
        return candidates

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
def _log_forward(np.ndarray[dtype_t, ndim=2] log_a,
                 np.ndarray[dtype_t, ndim=2] log_b,
                 context_trie,
                 log_c_tr_trie,
                 dict id_c,
                 np.ndarray[np.uint8_t, ndim=1] state_c,
                 np.ndarray[dtype_t, ndim=2] log_alpha):
    log_alpha[:] = -np.inf
    cdef int n = log_a.shape[0]
    cdef int n_contexts = log_a.shape[1]
    cdef int T = len(log_b)
    cdef int i, i_, t
    cdef str c, c_
    cdef list contexts = context_trie.keys()
    for i in range(n_contexts):
        t = 0
        log_alpha[t][i] = \
            context_trie[contexts[i]] + \
            log_b[t, state_c[i]]
    cdef np.ndarray[dtype_t, ndim=1] tmp = np.zeros(n_contexts)
    cdef dtype_t log_transition
    for t in range(T - 1):
        for i, c in enumerate(contexts):
            log_alpha[t+1, i] = np.log(0.)
            tmp.fill(np.log(0.))
            for c_ in get_list_c(context_trie, c[1:]):
                i_ = id_c[c_]
                log_transition = log_a[state_c[i], i_]
                if len(c_) > t:
                    log_transition = log_tr_p(c[0], c_[:t+1], context_trie,
                                              log_c_tr_trie, n)
                tmp[i_] = log_alpha[t, i_] + log_transition
            log_alpha[t+1, i] = _logsumexp(tmp) + log_b[t + 1, state_c[i]]


@cython.boundscheck(False)
def _log_backward(np.ndarray[dtype_t, ndim=2] log_a,
                  np.ndarray[dtype_t, ndim=2] log_b,
                  context_trie,
                  dict id_c,
                  np.ndarray[np.uint8_t, ndim=1] state_c,
                  np.ndarray [dtype_t, ndim=2] log_beta):
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
            log_beta[t, i] = _logsumexp(tmp)


@cython.boundscheck(False)
def _log_ksi(np.ndarray[dtype_t, ndim=2] log_a,
             np.ndarray[dtype_t, ndim=2] log_b,
             context_trie,
             dict id_c,
             np.ndarray[np.uint8_t, ndim=1] state_c,
             np.ndarray[dtype_t, ndim=2] log_alpha,
             np.ndarray[dtype_t, ndim=2] log_beta,
             np.ndarray[dtype_t, ndim=3] log_ksi):
    log_ksi.fill(np.log(0.))
    cdef int T = len(log_b)
    cdef int n = log_a.shape[0]
    cdef int n_contexts = log_a.shape[1]
    cdef int t, i, q, i_
    cdef str c, c_
    cdef np.ndarray[dtype_t] tmp = np.zeros(log_a.shape[1])
    for t in range(T - 1):
        for i, c in enumerate(context_trie.keys()):
            for q in range(n):
                tmp.fill(np.log(0.))
                for c_ in get_list_c(context_trie, str(q)+c):
                    i_ = id_c[c_]
                    tmp[i_] = log_alpha[t, i] + log_a[q, i] + \
                              log_b[t+1, q] + log_beta[t+1, i_]
                log_ksi[t, q, i] = _logsumexp(tmp)
    log_ksi -=  \
        _logsumexp_ksi_12(log_ksi).reshape((T, 1, 1))



def _test_logsumexp(X):
    start = time.time()
    logsumexp(X)
    time1=time.time()-start

    start = time.time()
    _logsumexp(X)
    time2=time.time()-start

    return time1-time2, (time1, time2)


def test_logsumexp_ksi(X):
    start = time.time()
    logsumexp(X, axis=(1,2))
    time1=time.time()-start

    start = time.time()
    _logsumexp_ksi_12(X)
    time2=time.time()-start

    return time1-time2, (time1, time2)

