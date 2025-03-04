import numpy as np

def calculate_loading(loading_logs, param_list=None):
    # Assume last dimension is the index for sequences
    arr_shape = loading_logs.shape
    if param_list is None:
        # Iterate over all dimensions except the last one
        load_rate = np.zeros(arr_shape[:-1])
        for idx in np.ndindex(arr_shape[:-1]):
            load_rate[idx] = np.mean(loading_logs[idx])
        nloads = arr_shape[-1]
    else: 
        unique_params = np.unique(param_list)
        nparams = len(unique_params)
        nloads = np.zeros(arr_shape[:-1] + (nparams,), dtype=int)
        load_rate = np.zeros(arr_shape[:-1] + (nparams,))
        for param_idx, param in enumerate(unique_params):
            mask = param_list == param
            for idx in np.ndindex(arr_shape[:-1]):
                load_rate[idx][param_idx] = np.mean(loading_logs[idx][mask])
                nloads[idx][param_idx] = np.sum(mask)
    load_rate_err = np.sqrt(load_rate * (1 - load_rate) / nloads)
    return load_rate, load_rate_err

def calculate_survival(loading_logs, survival_logs, param_list=None):
    # Assume last dimension is the index for sequences
    arr_shape = loading_logs.shape
    if param_list is None:
        # Iterate over all dimensions except the last one
        nloads = np.zeros(arr_shape[:-1], dtype=int)
        nsurvs = np.zeros(arr_shape[:-1], dtype=int)
        surv_prob = np.zeros(arr_shape[:-1])
        surv_prob_err = np.zeros(arr_shape[:-1])
        for idx in np.ndindex(arr_shape[:-1]):
            nloads[idx] = np.sum(loading_logs[idx])
            nsurvs[idx] = np.sum(survival_logs[idx][loading_logs[idx] > 0])
        raw_p = nsurvs / nloads
        surv_prob = (raw_p + 1/(2 * nloads))/(1 + 1 / nloads)
        surv_errs = np.sqrt(raw_p * (1 - raw_p) / nloads + 1/(4 * nloads**2)) / (1 + 1/nloads) 
    else: 
        unique_params = np.unique(param_list)
        nparams = len(unique_params)
        nloads = np.zeros(arr_shape[:-1] + (nparams,), dtype=int)
        nsurvs = np.zeros(arr_shape[:-1] + (nparams,), dtype=int)
        for param_idx, param in enumerate(unique_params):
            mask = param_list == param
            for idx in np.ndindex(arr_shape[:-1]):
                nloads[idx][param_idx] = np.sum(loading_logs[idx][mask])
                nsurvs[idx][param_idx] = np.sum(survival_logs[idx][mask & (loading_logs[idx] > 0)])
        raw_p = nsurvs / nloads
        surv_prob = (raw_p + 1/(2 * nloads))/(1 + 1 / nloads)
        surv_errs = np.sqrt(raw_p * (1 - raw_p) / nloads + 1/(4 * nloads**2)) / (1 + 1/nloads) 
    return surv_prob, surv_errs

def calculate_mean_value(loading_logs, vals, param_list=None):
    # Assume last dimension is the index for sequences
    arr_shape = loading_logs.shape
    if param_list is None:
        # Iterate over all dimensions except the last one
        mean_vals = np.zeros(arr_shape[:-1])
        mean_vals_errs = np.zeros(arr_shape[:-1])
        for idx in np.ndindex(arr_shape[:-1]):
            considered_vals = vals[idx][loading_logs[idx] > 0]
            mean_vals[idx] = np.mean(considered_vals)
            mean_vals_errs[idx] = np.std(considered_vals) / np.sqrt(len(considered_vals))
    else: 
        unique_params = np.unique(param_list)
        nparams = len(unique_params)
        mean_vals = np.zeros(arr_shape[:-1] + (nparams,))
        for param_idx, param in enumerate(unique_params):
            mask = param_list == param
            for idx in np.ndindex(arr_shape[:-1]):
                considered_vals = vals[idx][(loading_logs[idx] > 0) & mask]
                mean_vals[idx][param_idx] = np.mean(considered_vals)
                mean_vals_errs[idx][param_idx] = np.std(considered_vals) / np.sqrt(len(considered_vals))
    return mean_vals, mean_vals_errs