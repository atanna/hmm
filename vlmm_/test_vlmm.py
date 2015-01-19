import random
from vlmm_.vlmm import VLMM


def random_data(n, alphabet="abc"):
    return "".join([random.choice(alphabet) for i in range(n)])


def test():
    data = random_data(100)
    vlmm = VLMM().fit(data, k=3, min_num=2)
    contexts = list(vlmm.get_contexts(data))

    print("data:", data)
    print("contexts:", contexts)

test()