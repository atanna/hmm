import os
import random
import numpy as np
from chipseq.track import Track
from vlhmm_.multi_vlhmm import MultiVLHMM
from warnings import filterwarnings


def get_real_data(chr_i=1, bin_size=400, min_sample_len=4, max_n_samples=-1,
                  protein="H3K27ac"):
    data = np.genfromtxt(
        "resources/{}/chr{}_{}.txt".format(protein, chr_i, bin_size))
    arr_data = []
    t_start, t_fin = 0, 1
    sample_positions = []
    for i, x in enumerate(data):
        if x == 0:
            if t_fin - t_start > min_sample_len:
                arr_data.append(np.array(data[t_start: t_fin]))
                sample_positions \
                    .append(np.array(list(range(t_start, t_fin)))
                            .astype(np.int) * bin_size)
            t_start = t_fin
        t_fin += 1
    _T = len(arr_data)

    def sort_sample(arr_data, sample_positions):
        arr_data, sample_positions = zip(
            *sorted(zip(arr_data, sample_positions),
                    key=lambda x: len(x[0]), reverse=True))
        return np.array(arr_data), np.array(sample_positions)

    arr_data, sample_positions = sort_sample(arr_data, sample_positions)
    T = max_n_samples if max_n_samples > 0 else _T
    tail_T = int(0.6 * T)
    perm = np.random.permutation(_T - tail_T)
    arr_data[tail_T:] = np.array(arr_data[tail_T:])[perm]
    sample_positions[tail_T:] = np.array(sample_positions[tail_T:])[perm]
    arr_data, sample_positions = sort_sample(arr_data[:T],
                                             sample_positions[:T])
    return arr_data, sample_positions


def real_test(arr_data, max_len=4, th_prune=0.04,
              n=2, path="", sample_pos=None,
              data_params={},
              **fit_params):
    path = "{}/{}/".format(path, random.randrange(1e3))
    if not os.path.exists(path):
        os.makedirs(path)

    vlhmm = MultiVLHMM(n).fit(arr_data, max_len=max_len, th_prune=th_prune,
                              **fit_params)

    print(path)
    vlhmm.save_fitting_info(path=path, th_prune=th_prune,
                            data_params=data_params, **fit_params)

    states = vlhmm.get_hidden_states()

    track = Track(data_params.get("chr_i", 1), sample_pos, arr_data, path,
                  data_params.get("bin_size", 200))
    track.create_track_peaks("vlhmm", states, 1)
    track.create_track_data(int(10./9*vlhmm.emission.alpha[1]))
    track.create_track_data_bar(vlhmm.emission.alpha[1])


def run_real_test():
    max_len = 5

    chr_i = 4
    bin_size = 200
    protein = "H3K27ac"
    # protein="H3K4me3"

    data_params = dict(chr_i=chr_i, bin_size=bin_size,
                       min_sample_len=5,
                       max_n_samples=8000,
                       protein=protein)
    arr_data, sample_pos = get_real_data(**data_params)
    path = "test_results/{}/chr_{}/bin_size_{}/".format(protein, chr_i, bin_size)

    real_test(arr_data,
              path=path,
              max_len=max_len,
              sample_pos=sample_pos,
              data_params=data_params,
              max_log_p_diff=5.,
              parallel_params=dict(n_jobs=1))


if __name__ == "__main__":
    filterwarnings("ignore")
    run_real_test()
