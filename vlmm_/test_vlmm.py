import random
from vlmm_.vlmm import VLMM


def random_data(n, alphabet="abc"):
    return "".join([random.choice(alphabet) for i in range(n)])


def test():
    data = random_data(50, alphabet="ab")
    hmm_params = dict(n_starts=3,
                      log_eps=2e-3, max_iter=1e2)
    vlmm = VLMM().fit(data, type_vlmm="h_context", k=3, **hmm_params)

    print("data:", data)
    print("contexts:", vlmm.c)

    print("\nstates_contexts_hmm:", vlmm.n_contexts)
    sample = vlmm.sample(100)
    print(sample)
    print(vlmm.score(sample))

    n = vlmm.n_contexts
    print("\nhierarchy_hmm:", n)
    vlmm = VLMM(n).fit(data, k=3, min_num=2, **hmm_params)
    sample = vlmm.sample(100)
    print(sample)
    print(vlmm.score(sample))

test()

