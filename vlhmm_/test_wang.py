from vlhmm_.test import get_data
from vlhmm_.vlhmm_wang import VLHMMWang


def test_wang():
    data = get_data(100)
    # show_data(data)
    n = 3
    vlhmm = VLHMMWang(n)
    vlhmm.fit(data, max_len=3, n_iter=3)
    print(vlhmm.context_trie.contexts)


test_wang()