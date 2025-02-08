import numpy as np

# This module contains common operators we might want to measure
# The inputs are nimgs x nsites x nseqs
# The outputs are either a single value or a nparray. Errors are also calculated here.
def avg_n(arr):
    arr = arr[0,:,:]
    site_avg = np.mean(arr, axis=0)
    res = np.mean(site_avg)
    res_err = np.std(site_avg)
    return res, res_err

def n_i(idx):
    # Index is one indexed
    def n_i_fn(arr):
        arr = arr[0,:,:]
        per_seq = 1 - arr[idx - 1, :]
        res = np.mean(per_seq)
        res_err = np.std(per_seq)
        return res, res_err
    return n_i_fn
