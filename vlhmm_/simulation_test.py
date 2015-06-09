import numpy as np
from vlhmm_.multi_vlhmm import MultiVLHMM
from warnings import filterwarnings
from vlhmm_.vlhmm import VLHMM


def simulation_test(vlhmm, contexts, log_a, T=int(1e4),
               data=None, max_len=4,
               path="tests/",
               e_params={},
               **fit_params):

    real_e_params = "{}".format(e_params)
    if data is None:
        data, emission = vlhmm.sample_(T, contexts, log_a,
                                                **e_params)
        real_e_params = emission.get_str_params()

    vlhmm.fit(data, max_len=max_len, **fit_params)

    vlhmm.save_fitting_info(contexts, log_a, T=T,
                            path=path,
                            real_e_params=real_e_params,
                            **fit_params)
    print(path)


def run_comparison_test():
    T = int(1e4)
    n_parts = 8
    size_ = int(T/n_parts)

    contexts = ["00", "01", "1"]
    log_a = np.log(np.array(
        [[0.7, 0.4, 0.2],
         [0.3, 0.6, 0.8]]
    ))
    e_params = dict(alpha=[2, 15])

    data, emission = VLHMM.sample_(T, contexts, log_a, **e_params)

    arr_data = [data[i*size_: (i+1)*size_] for i in range(n_parts)]
    arr_T = list(map(len, arr_data))

    n = len(log_a)
    test_name = "comparison test"

    simulation_test(MultiVLHMM(n), contexts, log_a, T=arr_T,
               data=arr_data, max_len=4, e_params=e_params,
               parallel_params=dict(n_jobs=-1, backend='threading'),
               path="tests/{}/threading/".format(test_name))

    simulation_test(MultiVLHMM(n), contexts, log_a, T=arr_T,
               data=arr_data, max_len=4, e_params=e_params,
               parallel_params=dict(n_jobs=-1, backend='multiprocessing'),
               path="tests/{}/multiprocessing/".format(test_name))

    simulation_test(VLHMM(n), contexts, log_a, T=T,
               data=data, max_len=4, e_params=e_params,
               path="tests/{}/vlhmm/".format(test_name))


if __name__ == "__main__":
    filterwarnings("ignore")
    run_comparison_test()
