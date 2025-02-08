import SingleData
import numpy as np

class NaCsData:
    """
        Main class for manipulating data from the NaCs experiment
    """
    def __init__(self, date_strs: list, time_strs: list, prefix: list, param_map: list = None):
        """
            Constructor for a NaCsData object

            Args:
                date_strs: a list of strings for the date of each dataset.
                time_strs: a list of strings for the times of each dataset.
                prefix: a list of strings for the prefix of each dataset.
                param_map: a list of numpy arrays corresponding to what each parameter in each dataset should correspond to. Empty array applies no mapping to that particular dataset.
            Returns:
                TODO
            Raises:
                TODO
        """
        # ADD ERROR CHECKING
        if not isinstance(date_strs, list):
            date_strs = [date_strs]
        if not isinstance(time_strs, list):
            time_strs = [time_strs]
        n_datasets = len(date_strs)
        if not isinstance(prefix, list):
            prefix = [prefix for i in range(n_datasets)]
        temp_datasets = []
        for i in range(n_datasets):
            try:
                this_data = SingleData.SingleData(date_strs[i], time_strs[i], prefix[i])
            except:
                print('Error loading dataset ' + date_strs[i] + '_' + time_strs[i])
                raise
            temp_datasets.append(this_data)
        self.datasets = temp_datasets
        self.param_map = param_map
        self.n_datasets = n_datasets

    def get_logicals(self, img_idxs : list = None, seq_idxs: list = None):
        """
            Get combined logicals for all datasets according to the selected img_idxs and seq_idxs

            Args:
                img_idxs: list of indices of images desired. Each element  is an nparray corresponds to a dataset. A 0 indicates all images.
                          Can also be a single np array in which case it'll be applied to each dataset.
                          By default, concatenates all the images of each dataset.
                seq_idxs: list of indices to consider for each dataset. A 0 indicates to take all members of that dataset. An empty array indicates to not take any element from that dataset.
                          Can also be a single np array in which case it'll be applied to each dataset.
                          By default, concatenates all sequences of each dataset.
            Returns:
                The logicals and param_list. 
            Raises:
                TODO
        """
        if img_idxs == None:
            img_idxs = [0 for i in range(self.n_datasets)]
        if seq_idxs == None:
            seq_idxs = [0 for i in range(self.n_datasets)]
        if isinstance(img_idxs, np.ndarray):
            img_idxs = [img_idxs for i in range(self.n_datasets)]
        if isinstance(seq_idxs, np.ndarray):
            seq_idxs = [seq_idxs for i in range(self.n_datasets)]
        if seq_idxs != None and len(seq_idxs) != self.n_datasets:
            print("seq_idxs should be None or the length of it needs to match the number of datasets")
            return
        n_imgs = len(img_idxs[0])
        n_sites = self.datasets[0].num_sites
        log_array = np.zeros([n_imgs, n_sites, 0])
        param_list = np.zeros(0)
        for i in range(self.n_datasets):
            if seq_idxs[i] == []:
                continue
            elif (not isinstance(seq_idxs[i], np.ndarray)) and (not isinstance(img_idxs[i], np.ndarray)):
                these_logicals, this_param_list, this_seq_idxs = self.datasets[i].get_logicals()
            elif not isinstance(seq_idxs[i], np.ndarray):
                these_logicals, this_param_list, this_seq_idxs = self.datasets[i].get_logicals([], img_idxs[i])
            elif not isinstance(img_idxs[i], np.ndarray):
                these_logicals, this_param_list, this_seq_idxs = self.datasets[i].get_logicals(seq_idxs[i], [])
            else:
                these_logicals, this_param_list, this_seq_idxs = self.datasets[i].get_logicals(seq_idxs[i], img_idxs[i])
            if self.param_map != None and self.param_map[i] != []:
                this_param_list = self.param_map[i][this_param_list]
            log_array = np.concatenate((log_array, these_logicals), axis=2)
            param_list = np.concatenate((param_list, this_param_list), axis=0)
        return log_array, param_list

    def get_survival_prob_by_site(self, load_imgs: list, surv_imgs: list, seq_idxs: list = None, bWilsonScoreInterval: bool = True):
        """
            Get survival probability for all datasets according to the selected load_imgs, surv_imgs and seq_idxs per site

            Args:
                load_imgs: list of numpy arrays indicating which images are meant to be the loading images for each dataset
                surv_imgs: list of numpy arrays indicating which images are meant to be the survival images for each dataset
                seq_idxs: list of indices to consider for each dataset. A -1 indicates to take all members of that dataset. An empty array indicates to not take any element from that dataset.
                          Can also be a single np array in which case it'll be applied to each dataset.
                          By default, concatenates all sequences of each dataset.
                bWilsonScoreInterval: Uses WilsonScoreInterval to determine errorbar. https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
                
            Returns:
                The survival probability per parameter and site. First dimension is parameter and second is site. Also returns the symmetric errorbar and parameters.
            Raises:
                TODO
        """
        loading_log, surv_log, param_list = self._get_load_and_surv_logs(load_imgs, surv_imgs, seq_idxs)
        nsites = loading_log.shape[0]
        unique_params = np.unique(param_list)
        survs = np.zeros([len(unique_params), nsites])
        surv_errs = np.zeros([len(unique_params), nsites])
        for idx, param in enumerate(unique_params):
            param_idxs = param_list == param
            loads_by_site = np.sum(loading_log[:,param_idxs], axis=1)
            survs_by_site = np.sum(np.logical_and(loading_log[:,param_idxs], surv_log[:, param_idxs]), axis=1)
            p = survs_by_site / loads_by_site
            if bWilsonScoreInterval:
                survs[idx, :] = (p + 1/(2 * loads_by_site))/(1 + 1 / loads_by_site)
                surv_errs[idx, :] = np.sqrt(p * (1 - p) / loads_by_site + 1/(4 * loads_by_site**2)) / (1 + 1/loads_by_site) 
            else:
                survs[idx, :] = p
                surv_errs[idx, :] = np.sqrt(p * (1 - p) / loads_by_site)
        return survs, surv_errs, unique_params

    def get_survival_prob_avg(self, load_imgs: list, surv_imgs: list, site_idxs: np.ndarray = None, seq_idxs: list = None, bWilsonScoreInterval: bool = True):
        """
            Get average survival probability for all datasets according to the selected load_imgs, surv_imgs, site_idxs and seq_idxs per site

            Args:
                load_imgs: list of numpy arrays indicating which images are meant to be the loading images for each dataset
                surv_imgs: list of numpy arrays indicating which images are meant to be the survival images for each dataset
                site_idxs: numpy array of sites to average.
                seq_idxs: list of indices to consider for each dataset. A -1 indicates to take all members of that dataset. An empty array indicates to not take any element from that dataset.
                          Can also be a single np array in which case it'll be applied to each dataset.
                          By default, concatenates all sequences of each dataset.
                bWilsonScoreInterval: Uses WilsonScoreInterval to determine errorbar. https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
                
            Returns:
                The survival probability per parameter and site. First dimension is parameter and second is site.
            Raises:
                TODO
        """
        loading_log, surv_log, param_list = self._get_load_and_surv_logs(load_imgs, surv_imgs, seq_idxs)
        nsites = loading_log.shape[0]
        unique_params = np.unique(param_list)
        if site_idxs == None:
            site_idxs = np.array([i for i in range(nsites)])
        else:
            site_idxs = site_idxs - 1
        survs = np.zeros(len(unique_params))
        surv_errs = np.zeros(len(unique_params))
        for idx, param in enumerate(unique_params):
            param_idxs = param_list == param
            loads_tot = np.sum(loading_log[site_idxs][:, param_idxs])
            survs_tot = np.sum(np.logical_and(loading_log[site_idxs][:, param_idxs], surv_log[site_idxs][:, param_idxs]))
            p = survs_tot / loads_tot
            if bWilsonScoreInterval:
                survs[idx] = (p + 1/(2 * loads_tot))/(1 + 1 / loads_tot)
                surv_errs[idx] = np.sqrt(p * (1 - p) / loads_tot + 1/(4 * loads_tot**2)) / (1 + 1/loads_tot) 
            else:
                survs[idx] = p
                surv_errs[idx] = np.sqrt(p * (1 - p) / loads_tot)
        return survs, surv_errs, unique_params

    def get_load_idxs(self, load_imgs: list, load_condition = None, seq_idxs: list = None):
        """
            Calculate the indices of sequences where load_condition evaluates to a nonzero value

            Args:
                load_imgs: list of nparrays that define the input to the load_condition callable per dataset
                load_condition: A callable that acts on logicals of the format nimgs x nsites x nseqs. If load_condition is a np.ndarray, then it is interpretted as site idxs
            
            Returns:
                The idxs and params of successful loads. Note that returned idxs are not absolute sequence indices but relative to those from seq_idxs
        """
        if isinstance(load_imgs, np.ndarray):
            load_imgs = [load_imgs for i in range(self.n_datasets)]
        if isinstance(seq_idxs, np.ndarray):
            seq_idxs = [seq_idxs for i in range(self.n_datasets)]
        log_array, param_list = self.get_logicals(load_imgs, seq_idxs)
        if load_condition is None:
            nsites = log_array.shape[1]
            load_condition = np.ndarray([i + 1 for i in range(nsites)])
        if isinstance(load_condition, np.ndarray):
            abs_site_idxs = np.abs(load_condition) - 1
            ret_logs = log_array[:,abs_site_idxs,:]
            if any(load_condition < 0):
                mod_arr = ret_logs[:,np.where(load_condition < 0),:]
                ret_logs[:,np.where(load_condition < 0),:] = np.where(mod_arr == 0, 1, 0)
                calculated_loads = np.all(ret_logs, axis=(0,1))
        else:
            calculated_loads = load_condition(log_array)
        # Return of above function should be single array for each sequence
        idxs = np.nonzero(calculated_loads)
        params = param_list[idxs]
        return idxs, params

    def get_expectation_val(self, load_imgs: list, surv_imgs: list, load_condition, observable: list, seq_idxs: list = None):
        """
            Calculate the expectation value of observable, which is a callable (i.e. function) given the load_condition for load_imgs.
            surv_imgs define the input to the observable

            Args:
                load_imgs: list of nparrays that define the input to the load_condition callable per dataset
                surv_imgs: list of nparrays that define the input to the observable callable per dataset
                load_condition: callable that acts on the images from load_imgs. The input is nimgs x nsites x nseqs
                observable: list of callables for each dataset that acts on the images from surv_imgs. The input is nimgs x nsites x nseqs. The output can either be just a number per sequence, or a list per sequence.
            Returns:
                A list of lists, where the outer nesting is per observable and the inner is per parameter
            Raises:

        """
        if not isinstance(observable, list):
            observable = [observable]
        idxs, param_list = self.get_load_idxs(load_imgs, load_condition, seq_idxs)
        surv_logs, _ = self.get_logicals(surv_imgs, seq_idxs)
        # Cut down the indices to consider for surv_imgs
        surv_logs = surv_logs[:,:,idxs]
        unique_params = np.unique(param_list)
        n_obs = len(observable)
        res = []
        res_err = []
        for i in range(n_obs):
            this_obs = observable[i]
            temp_res = []
            temp_res_err = []
            for idx, param in enumerate(unique_params):
                param_idxs = param_list == param
                this_result, this_result_err = this_obs(surv_logs[:,:, param_idxs])
                temp_res.append(this_result)
                temp_res_err.append(this_result_err)
            res.append(temp_res)
            res_err.append(temp_res_err)
        return res, res_err, unique_params

    def _get_load_and_surv_logs(self, load_imgs: list, surv_imgs: list, seq_idxs: list = None):
        """
            Private function to get load and survival logicals as indicated by load_imgs, surv_imgs and seq_idxs

            Args:
                load_imgs: list of numpy arrays indicating which images are meant to be the loading images for each dataset
                surv_imgs: list of numpy arrays indicating which images are meant to be the survival images for each dataset
                seq_idxs: list of indices to consider for each dataset. A -1 indicates to take all members of that dataset. An empty array indicates to not take any element from that dataset.
                          Can also be a single np array in which case it'll be applied to each dataset.
                          By default, concatenates all sequences of each datasets
                
            Returns:
                The loading logicals, survival logicals and param list
            Raises:
                TODO
        """
        if isinstance(load_imgs, np.ndarray):
            load_imgs = [load_imgs for i in range(self.n_datasets)]
        if isinstance(seq_idxs, np.ndarray):
            seq_idxs = [seq_idxs for i in range(self.n_datasets)]
        log_array, param_list = self.get_logicals(load_imgs, seq_idxs)
        loading_log = np.all(log_array==1, axis=0)
        surv_array, param_list = self.get_logicals(surv_imgs, seq_idxs)
        surv_log = np.all(surv_array==1, axis=0)
        return loading_log, surv_log, param_list