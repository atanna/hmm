from numpy import array

from hmm_.hmm import HMM


def test(data, n_states_from=2, n_states_to=8):
    print("data  ", data)
    for i in range(n_states_from, n_states_to):
        p, model = HMM(i).optimal_model(data, n_starts=5,
                                        eps=1e-17, max_iter=1e2)
        print("hmm({}): {}".format(i, p))
        if abs(p-1.) < 1e-8:
            break
    print("states", HMM(i).optimal_state_sequence(model, data))
    print("model\n{}\n".format(model))


def main():
    test(array([0, 1, 2, 2, 1, 0, 0, 1, 2, 2, 1, 0, 0, 1, 2]))
    test(array([0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1]))

main()

