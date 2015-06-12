import os
import numpy as np
from vlhmm_.context_tr_trie import ContextTransitionTrie


def simulation_test(contexts, log_a, n=2, T=int(5e3), max_len=4,
                    th_prune=0.007, path="test_results/tries_test/",
                    save_results=True, log_c_p=None):
    real_trie = ContextTransitionTrie(n=n)
    real_trie.recount_with_log_a(log_a, contexts, log_c_p)
    data = real_trie.sample(T)

    trie = ContextTransitionTrie(data, max_len=max_len)

    trie.prune(th_prune)

    if save_results:
        if not os.path.exists(path):
            os.makedirs(path)

        fname_real = "{}real_trie.jpg".format(path)
        fname_predict = "{}predicted_trie.jpg".format(path)
        real_trie.draw(fname_real)
        trie.draw(fname_predict)
        print(path)


if __name__ == "__main__":
    T = int(4e4)

    contexts = ["00", "010", "011", "1"]
    log_a = np.log(np.array([
        [0.8, 0.6, 0.1, 0.2],
        [0.2, 0.4, 0.9, 0.8]]))
    max_len = 4
    simulation_test(contexts, log_a, max_len=max_len, T=T,
                    path="test_results/tries_test/")
