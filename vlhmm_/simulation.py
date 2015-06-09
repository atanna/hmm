import os
import numpy as np
from multi_vlhmm import MultiVLHMM
from warnings import filterwarnings
from vlhmm import VLHMM


def save_fitting_info(vlhmm, contexts=None, log_a=None, path="",
                      lang_plot='ru', **kwargs):
    if not os.path.exists(path):
        os.makedirs(path)
    vlhmm.create_img(contexts, log_a, path, language=lang_plot)

    with open("{}info.txt".format(path), "wt") as f:
        f.write("params:\n")
        f.write("{}\n\n".format(kwargs))
        f.write("real_params:\ncontexts: {}\na:\n{}\nemission: {}\n\n"
                .format(contexts, np.round(np.exp(log_a), 3),
                        kwargs.get("real_e_params", "unknown")))
        f.write("fitting:\n")
        for i, info in enumerate(vlhmm.info):
            f.write("EM_{}:\n{}\n".format(i, info))
        f.write("\n")
        f.write("fdr, fndr: {}\n\n".format(vlhmm.estimate_fdr_fndr()))
        f.write("fitting time: {}\n".format(vlhmm.fit_time))


def simulation(vlhmm, contexts, log_a, T=int(1e4),
               data=None, max_len=4,
               path="tests/",
               e_params={},
               **fit_params):

    real_e_params = "{}".format(e_params)
    if data is None:
        data, emission = vlhmm.sample_(T, contexts, log_a,
                                                **e_params)
        real_e_params = emission.get_str_params()

    vlhmm.fit(data, **fit_params)

    save_fitting_info(vlhmm, contexts, log_a, T=T,
                      max_len=max_len, path=path,
                      real_e_params=real_e_params,
                      **fit_params)
    print(path)


def go_sample_test():
    T = int(1e5)
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

    simulation(MultiVLHMM(n), contexts, log_a, T=arr_T,
               data=arr_data, max_len=4, e_params=e_params,
               parallel_params=dict(n_jobs=-1, backend='threading'),
               path="tests/{}/threading/".format(test_name))

    simulation(MultiVLHMM(n), contexts, log_a, T=arr_T,
               data=arr_data, max_len=4, e_params=e_params,
               parallel_params=dict(n_jobs=-1, backend='multiprocessing'),
               path="tests/{}/multiprocessing/".format(test_name))

    simulation(VLHMM(n), contexts, log_a, T=T,
               data=data, max_len=4, e_params=e_params,
               path="tests/{}/vlhmm/".format(test_name))


if __name__ == "__main__":
    filterwarnings("ignore")
    go_sample_test()
