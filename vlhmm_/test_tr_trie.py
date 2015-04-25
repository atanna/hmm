import os
import random
import numpy as np
import pylab as plt
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


def test_context_tr_trie_sample(contexts, log_a, n=2, T=int(5e3), max_len=4, th_prune=3e-2, path="tries/tests/", draw=False, log_c_p=None):

    data = ContextTransitionTrie.sample_(T, contexts, log_a)

    c_tr_trie = ContextTransitionTrie(n=n)
    c_tr_trie.recount_with_log_a(log_a, contexts, log_c_p)
    print("{} contexts: {}".format(c_tr_trie.n_contexts, c_tr_trie.seq_contexts))


    trie = ContextTransitionTrie(data, max_len=max_len)
    print("{}  {}".format(trie.n_contexts, trie.seq_contexts))

    print(trie.contexts.items())

    while trie.prune(th_prune):
        print("prune: {}  {}".format(trie.n_contexts, trie.seq_contexts))

    for c in trie.seq_contexts:
        for q in trie.alphabet:
            print("p({}|{})={}   {}".format(q, c, trie.tr_p(q, c), c_tr_trie.tr_p(q, c)))
    print()

    for c in contexts:
        for q in trie.alphabet:
            print("p({}|{})={}   {}".format(q, c, trie.tr_p(q, c), c_tr_trie.tr_p(q, c)))
    print()

    print(trie.seq_contexts)
    print("a:\n{}".format(np.exp(trie.count_log_a())))
    print("c_p:\n{}".format(np.exp(list(zip(*trie.contexts.items()))[1])))

    if draw:
        if not os.path.exists(path):
                os.makedirs(path)
        rand_number = random.randrange(100)
        fname = "{}{}_{}_{}_{}".format(path, T, th_prune, max_len, rand_number)

        fname_real = "{}real.jpg".format(fname)
        fname_predict = "{}predict.jpg".format(fname, rand_number)
        c_tr_trie.draw(fname_real)
        trie.draw(fname_predict)
        print()
        print(fname)
        fig = plt.figure()
        ax = fig.add_subplot("111")
        ax.set_title("Real tree")
        ax.imshow(plt.imread(fname_real))
        plt.show()
        fig = plt.figure()
        ax = fig.add_subplot("111")
        ax.set_title("Predicted tree")
        ax.imshow(plt.imread(fname_predict))
        plt.show()


def prune_test(contexts, log_a, th_prune=0.01, n=2, log_c_p=None):
    trie = ContextTransitionTrie(n=n)
    trie.recount_with_log_a(log_a, contexts, log_c_p)
    print("{} contexts: {}".format(trie.n_contexts, trie.seq_contexts))


    while trie.prune(th_prune):
        print("prune: {}  {}".format(trie.n_contexts, trie.seq_contexts))


    print(trie.seq_contexts)
    print("a:\n{}".format(np.exp(trie.count_log_a())))
    print("c_p:\n{}".format(np.exp(list(zip(*trie.contexts.items()))[1])))

    fname = "1.jpg"
    trie.draw(fname)
    plt.imshow(plt.imread(fname))
    os.remove(fname)
    plt.show()



def test_():
    contexts = ["000", "0010", "0011", "01", "1"]
    log_a = np.log(np.array([
        [0.9, 0.2, 0.4, 0.3, 0.9],
        [0.1, 0.8, 0.6, 0.7, 0.1]
    ]
    ))
    log_c_p=None

  #   contexts =  ['0', '1000', '1001', '1010', '1011', '110', '1110', '1111']
  #   log_a = np.log(np.array(
  #       [[0.9231,  0.783,   1.,      0.5149,  0.,      0.5248,  0.5821,  1.],
  #        [0.0769,  0.217,   0.,      0.4851,  1.,      0.4752,  0.4179,  0.]]))
  #   log_c_p = np.log([ 0.8771494,   0.0177028,   0.01785328,  0.01749774,  0.01704328,  0.0301244,
  # 0.01554765,  0.00708146])

    contexts = ['00', '010', '011', '1']
    log_a = np.log(np.array(
        [[0.9462,  0.5248,  1.,     0.7132],
         [0.0538,  0.4752,  0.,      0.2868]]))

    contexts = ["00", "01", "1"]
    log_a = np.log(np.array([[0.7, 0.4, 0.2], [0.3, 0.6, 0.8]]
    ))

    contexts = ['00', '010', '011', '1']
    log_c_p = np.log([0.84, 0.03, 0.03, 0.09])
    # log_c_p = np.log([0.25, 0.25, 0.25, 0.25])
    log_a = np.log(
        [[0.95, 0.31,  1., 0.72],
         [0.05, 0.69,  0., 0.28]])

    contexts = ['0000', '0001', '001', '010', '011', '1']
    log_a = np.log([[0.968,   1.,      0.9276,  0.2673,  1.,      0.6765],
                    [ 0.032,   0.,      0.0724,  0.7327,  0.,      0.3235]])
    log_c_p = np.log([0.78481368,  0.03513084,  0.0376413,   0.02977187,  0.02910418,  0.08353813])


    # contexts = ['000', '001', '010', '011', '1']
    # log_a = np.log([[ 0.96937105,  0.9276,      0.2673 ,     1.,          0.6765    ],
    #  [ 0.03062895,  0.0724,      0.7327 ,     0. ,         0.3235    ]])
    # log_c_p = np.log([ 0.81994452,  0.0376413,   0.02977187,  0.02910418,  0.08353813])

    contexts = ['0000', '0001', '001', '010', '011', '1']
    log_a = np.log([[0.9629,  1.,      1.,      0.2534 , 1.,      0.7   ],
                    [0.0371,  0.,      0. ,     0.7466,  0.,      0.3   ]])
    log_c_p = np.log([ 0.80787515,  0.03095295,  0.03301776,  0.02632742,  0.02565991,  0.07616681])

    contexts = ['000', '001', '010', '011', '1']
    log_c_p = np.log([0.81, 0.04, 0.03, 0.03, 0.09])
    log_a = np.log(
        [[0.95,  1.,  0.34,  1.,  0.74],
         [0.05,  0.,  0.66,  0.,  0.26]])

    prune_test(contexts, log_a, log_c_p=log_c_p, th_prune=0.02)

    T = int(1e4)
    # T = 407
    max_len = 4
    # test_context_tr_trie_sample(contexts, log_a, max_len=max_len, T=T, th_prune=0.01, log_c_p=log_c_p, draw=True)
    # test_context_tr_trie_sample(contexts, log_a, max_len=6, T=int(1e4), th_prune=2e-2)
test_()