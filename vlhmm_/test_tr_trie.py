import random
import numpy as np
from vlhmm_.context_tr_trie import ContextTransitionTrie


def random_string(n, alphabet="abc"):
    return "".join([random.choice(alphabet) for i in range(n)])


def print_all_p(items):
        for s, val in items:
            print("p({}|{}) = {}".format(s[0], s[1:], np.exp(val)))


def test_context_tr_trie_with_data():

    data = random_string(100, "10")
    print(data)
    tr_prune = 1e-3
    context_transition_trie = ContextTransitionTrie(data)
    print(context_transition_trie.seq_contexts)
    context_transition_trie.prune(tr_prune)
    print("prune", context_transition_trie.seq_contexts)
    print(context_transition_trie.log_c_tr_trie.items())


def test_context_tr_trie_sample():
    alphabet = "01"

    contexts = ["00", "01", "10", "110", "111"]
    log_a = np.log(np.array(
        [[0.8, 0.4, 0.3, 0.2, 0.9],
         [0.2, 0.6, 0.7, 0.8, 0.1]]
    ))

    n, T = 2, int(5e3)
    data = ContextTransitionTrie.sample_(T, contexts, log_a)

    c_tr_trie = ContextTransitionTrie(n=2)
    c_tr_trie.recount_with_log_a(log_a, contexts)
    print("contexts:", c_tr_trie.seq_contexts)


    trie = ContextTransitionTrie(data, max_len=6)
    th_prune = 4e-2
    while trie.prune(th_prune):
        print("prune", trie.seq_contexts)

    for c in trie.seq_contexts:
        for q in alphabet:
            print("p({}|{})={}   {}".format(q, c, trie.tr_p(q, c), c_tr_trie.tr_p(q, c)))
    print()


test_context_tr_trie_sample()