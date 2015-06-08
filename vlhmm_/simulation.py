import os
import numpy as np
from scipy.stats.mstats_basic import mquantiles
from emission import PoissonEmission
from multi_vlhmm import MultiVLHMM
from warnings import filterwarnings
from vlhmm import VLHMM


def data_to_file(data, f_name):
    np.savetxt(f_name, data)


def data_from_file(f_name):
    return np.genfromtxt(f_name)


def get_parts(data, boards_parts=[0., 0.5, 0.75, 1.]):
    T = len(data)

    t = mquantiles(list(range(T + 1)), boards_parts)
    print(t)
    N = len(t)
    return [data[t[i - 1]:t[i]] for i in range(1, N)]


def go_vlhmm(vlhmm, data, contexts, log_a, path="", T=None,
             real_e_params="unknown", max_len=4, lang_plot='ru', **kwargs):
    vlhmm.fit(data, max_len=max_len, **kwargs)
    vlhmm.create_img(contexts, log_a, path, language=lang_plot)

    with open("{}info.txt".format(path), "wt") as f:
        f.write("params:\n")
        f.write("T={}  max_len={}\n".format(T, max_len))
        f.write("{}\n\n".format(kwargs))
        f.write("real_params:\ncontexts: {}\na:\n{}\nemission: {}\n\n"
                .format(contexts, np.round(np.exp(log_a), 3),
                        real_e_params))
        f.write("fitting:\n")
        for i, info in enumerate(vlhmm.info):
            f.write("EM_{}:\n{}\n".format(i, info))
        f.write("\n")
        f.write("fdr, fndr: {}\n\n".format(vlhmm.estimate_fdr_fndr()))
        f.write("fitting time: {}\n".format(vlhmm.fit_time))


def simulation(contexts, log_a, T=int(2e3), max_len=4, th_prune=0.01,
               log_pr_thresh=0.01, path="tests/vlhmm/",
               type_e="Poisson", start="k-means",
               save_data=False, data=None,
               start_params=None, lang_plot='ru',
               **e_params):
    n = len(log_a)
    real_e_params = "unknown"
    if data is None:
        data, emission = VLHMM.sample_(T, contexts, log_a,
                                       type_emission=type_e, **e_params)
        real_e_params = emission.get_str_params()

    if not os.path.exists(path):
        os.makedirs(path)

    if save_data:
        data_to_file(data, path + ".txt")

    go_vlhmm(VLHMM(n), data, contexts, log_a, path=path,
             T=T,
             real_e_params=real_e_params, max_len=max_len, start=start,
             th_prune=th_prune,
             log_pr_thresh=log_pr_thresh, type_emission=type_e,
             lang_plot=lang_plot,
             start_params=start_params)
    print(path)


def multi_simulation(contexts, log_a, T=int(1e3), arr_T=None,
                     arr_data=None, max_len=4,
                     th_prune=0.007, log_pr_thresh=0.05, n_parts=10,
                     max_log_p_diff=1.5, type_e="Poisson",
                     start="k-means", save_data=False,
                     path="tests/multi",
                     start_params=None,
                     lang_plot="ru",
                     **e_params):
    n = len(log_a)
    real_e_params = "unknown"
    if arr_data is None:
        if arr_T is None:
            arr_T = [int(T / n_parts)] * n_parts
        emission = PoissonEmission(n_states=n)
        emission._set_params(**e_params)
        arr_data, emission = zip(*[VLHMM.sample_(T, contexts, log_a,
                                                 type_emission=type_e,
                                                 emission=emission)
                                   for T in arr_T])
        emission = emission[0]
        real_e_params = emission.get_str_params()

    if not os.path.exists(path):
        os.makedirs(path)

    if save_data:
        data_to_file(arr_data, path + "multi_data.txt")

    T = [len(d) for d in arr_data]

    vlhmm = MultiVLHMM(n)
    go_vlhmm(vlhmm, list(arr_data), contexts, log_a, path=path, T=T,
             real_e_params=real_e_params, max_len=max_len, start=start,
             th_prune=th_prune,
             log_pr_thresh=log_pr_thresh, type_emission=type_e,
             max_log_p_diff=max_log_p_diff,
             start_params=start_params,
             lang_plot=lang_plot)
    print(path)


def go_sample_test():
    T = int(1e4)
    n_parts = int(T / 1000)

    contexts = ["00", "01", "1"]
    log_a = np.log(np.array(
        [[0.7, 0.4, 0.2],
         [0.3, 0.6, 0.8]]
    ))

    multi_simulation(contexts, log_a, T=T, max_len=4,
                     n_parts=n_parts,
                     path="tests/multi/")

    simulation(contexts, log_a, T=T, max_len=4,
               path="tests/vlhmm/")


if __name__ == "__main__":
    filterwarnings("ignore")
    go_sample_test()
