from vlhmm_.test import get_data
from vlhmm_.vlhmm_dumont import VLHMMDumont


def test_dumont():
    data = get_data(100)
    n=3
    vlhmm = VLHMMDumont(n)
    vlhmm.fit(data, max_len=4)
    print(vlhmm.context_trie.contexts)

test_dumont()