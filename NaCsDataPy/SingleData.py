import os
import h5py
import numpy as np

import utils
import DataProcessTools
import PlotProcessTools
from IPython.core.debugger import set_trace

class SingleData:
    """
        Class managing the IO with saved data files for the NaCs experiments.
    """
    def __init__(self, datestr: str, timestr: str, prefix: str):
        """
            Constructor for a SingleData object

            Args:
                TODO
            Returns:
                TODO
            Raises:
                TODO

        """
        # Allow for a full path specification
        if datestr.endswith('.mat'):
            self.master_fname = datestr
        else:
            dirstr = 'data_' + datestr + '_' + timestr
            filestr = dirstr + '.mat'
            full_filestr = os.path.join(datestr, dirstr)
            full_filestr = os.path.join(full_filestr, filestr)
            self.master_fname = os.path.join(prefix, full_filestr)
        print('Loading ' + self.master_fname)
        try:
            self.data_file : h5py._hl.files.File = h5py.File(self.master_fname)
        except FileNotFoundError:
            print('File not found.')
            raise
        except:
            print('Unknown error when opening master file')
            raise
        try:
            self.names_file : h5py._hl.files.File = h5py.File(self.master_fname[0:-4] + "_names.mat")
        except FileNotFoundError:
            print('Names file not found.')
            raise
        except:
            print('Unknown error when opening names file')
            raise

        self.data = utils.cvt_to_dict(self.data_file)
        
        if self.data['Scan']['version'] != 4:
            print('Only version 4 supported for now')
            raise
        self.num_sites = int(self.data['Scan']['NumSites'])
        self.num_imgs = int(self.data['Scan']['NumImages'])
        self.num_seqs = int(self.data['Analysis']['SummaryData']['n_seq'])
        self.signals = None # numpy array
        self.logicals = None # numpy array
        self.imgs = None # numpy array
        self.seq_idxs = None # 1 indexed
        self.seq_idxs_img = None # 1 indexed
        self.param_list = None
        self.img_fnames = utils.cvt_to_str(self.names_file, 'names/img_fnames')
        self.summary_fnames = utils.cvt_to_str(self.names_file, 'names/summary_fnames')
        self.log_files_cache = None # tracks which sequences are in which files
        self.img_files_cache = None


    def __del__(self):
        """ 
            Function for when the object is deleted. We need to make sure the file is closed.
        """
        self.data_file.close()
        self.names_file.close()

    def get_logicals(self, img_idxs=None, site_idxs =None, seq_idxs=None):
        # seq_idxs are sequence indices, 1 indexed
        # img_idxs are the images that are wanted. A negative number indicates that you want to apply the logical NOT.

        img_idxs = utils.cvt_arg(img_idxs, self.num_imgs)
        site_idxs = utils.cvt_arg(site_idxs, self.num_sites)
        seq_idxs = utils.cvt_arg(seq_idxs, self.num_seqs)

        self._load_logicals(seq_idxs)
        
        ret_logs = self.logicals.copy()
        # Modify imgs and sites
        if any(img_idxs < 0):
            act_img_idxs = np.abs(img_idxs[img_idxs < 0]) - 1
            mod_arr = ret_logs[act_img_idxs,:,:]
            ret_logs[act_img_idxs,:,:] = np.where(mod_arr == 0, 1, 0)
        if any(site_idxs < 0):
            act_site_idxs = np.abs(site_idxs[site_idxs < 0]) - 1
            mod_arr = ret_logs[:,act_site_idxs,:]
            ret_logs[:,act_site_idxs,:] = np.where(mod_arr == 0, 1, 0)
        idxs =  np.array([np.nonzero(self.seq_idxs == x)[0][0] for x in seq_idxs]) # Determine proper indices in self.seq_idxs for the requested ones

        return ret_logs[np.ix_(np.abs(img_idxs) - 1, np.abs(site_idxs) - 1, idxs)], self.param_list[idxs], self.seq_idxs[idxs]
        
    def get_signals(self, img_idxs=None, site_idxs =None, seq_idxs=None):
        # seq_idxs are sequence indices, 1 indexed
        # img_idxs are the images that are wanted.

        img_idxs = utils.cvt_arg(img_idxs, self.num_imgs)
        site_idxs = utils.cvt_arg(site_idxs, self.num_sites)
        seq_idxs = utils.cvt_arg(seq_idxs, self.num_seqs)

        self._load_logicals(seq_idxs) # this will load the signals if they are not loaded
        
        idxs =  np.array([np.nonzero(self.seq_idxs == x)[0][0] for x in seq_idxs])
        
        return self.signals[np.ix_(img_idxs - 1, site_idxs - 1, idxs)], self.param_list[idxs], self.seq_idxs[idxs]
    
    def get_imgs(self, img_idxs=None, seq_idxs=None):
        # img_idxs are the images that are wanted, one indexed
        # seq_idxs are sequence indices, 1 indexed.

        img_idxs = utils.cvt_arg(img_idxs, self.num_imgs)
        seq_idxs = utils.cvt_arg(seq_idxs, self.num_seqs)

        self._load_imgs(seq_idxs) # this will load the images if they are not loaded
        self._load_logicals(seq_idxs) # this will load the param list

        idxs =  np.array([np.nonzero(self.seq_idxs_img == x)[0][0] for x in seq_idxs])
        return self.imgs[:,:,img_idxs-1,:][:,:,:,idxs], self.param_list[idxs], self.seq_idxs_img[idxs]
    
    def get_avg_img_by_param(self, img_idxs=None, seq_idxs=None):
        # Get average image by parameter

        imgs, param_list, seq_idxs = self.get_imgs(img_idxs=img_idxs, seq_idxs=seq_idxs)
        unique_params = np.unique(param_list)
        nparams = len(unique_params)
        arr_shape = imgs.shape
        mean_imgs = np.zeros(arr_shape[:-1] + (nparams,))
        for param_idx, param in enumerate(unique_params):
            mask = param_list == param
            mean_imgs[:,:,:,param_idx] = np.mean(imgs[:,:,:,mask], axis=3)
        return mean_imgs

    def get_signals_by_param(self, img_idxs=None, site_idxs=None, seq_idxs=None):
        # Get signals by parameter

        sigs, param_list, seq_idxs = self.get_signals(img_idxs=img_idxs, site_idxs=site_idxs, seq_idxs=seq_idxs)
        unique_params = np.unique(param_list)
        ret_sigs = []
        for param_idx, param in enumerate(unique_params):
            mask = param_list == param
            ret_sigs.append(sigs[:,:,mask])
        return ret_sigs

    def get_site_by_site_survival(self, load_imgs, surv_imgs, site_idxs=[], seq_idxs=[]):
        # load_imgs are which images to be used for loading
        # surv_imgs are which images to be used for survival
        load_logicals, param_list,_ = self.get_logicals(img_idxs=load_imgs, seq_idxs=seq_idxs)
        surv_logicals, _, _ = self.get_logicals(img_idxs=surv_imgs, seq_idxs=seq_idxs)
        if len(site_idxs) == 0:
            site_idxs =  np.array([i + 1 for i in range(self.num_sites)])
        if isinstance(site_idxs, list):
            site_idxs = np.array(site_idxs)
        site_idxs = site_idxs - 1
        load_logicals = np.squeeze(np.all(load_logicals[:, site_idxs, :], axis=0))
        surv_logicals = np.squeeze(np.all(surv_logicals[:, site_idxs, :], axis=0))
        return DataProcessTools.calculate_survival(load_logicals, surv_logicals, param_list)
    
    def get_avg_survival(self, load_imgs, surv_imgs, site_idxs=[], seq_idxs=[]):
        load_logicals, param_list, these_seq_idxs = self.get_logicals(img_idxs=load_imgs, seq_idxs=seq_idxs)
        surv_logicals, _, _ = self.get_logicals(img_idxs=surv_imgs, seq_idxs=seq_idxs)
        if len(site_idxs) == 0:
            site_idxs =  np.array([i + 1 for i in range(self.num_sites)])
        if isinstance(site_idxs, list):
            site_idxs = np.array(site_idxs)
        site_idxs = site_idxs - 1
        load_logicals = np.squeeze(np.all(load_logicals[:, site_idxs, :], axis=0))
        surv_logicals = np.squeeze(np.all(surv_logicals[:, site_idxs, :], axis=0))
        load_logicals = np.squeeze(np.reshape(load_logicals, (1, len(site_idxs) * len(these_seq_idxs))))
        surv_logicals = np.squeeze(np.reshape(surv_logicals, (1, len(site_idxs) * len(these_seq_idxs))))
        param_list = np.tile(param_list, len(site_idxs))
        return DataProcessTools.calculate_survival(load_logicals, surv_logicals, param_list)
    
    def get_site_conditioned_survival(self, load_imgs, load_site_idxs, surv_imgs, surv_site_idxs, seq_idxs=[]):
        # load_imgs is which imgs to consider for loading
        # load_site_idxs is a list of lists, where each item are the sites to consider for that img
        # surv_imgs is which imgs to consider for survival
        # surv_site_idxs is a list of lists, where each item are the sites to consider for that img
        # For example, load_imgs = [1,2] and load_site_idxs = [[12, 14], [16, -18]] says consider cases where 
        # sites 12 and 14 are loaded on image 1 and site 16 is loaded on image 2, AND site 18 is not loaded on image 2
        
        # Start by flattening load_site_idxs and surv_site_idxs
        # all_logicals, param_list, _ = self.get_logicals(seq_idxs=seq_idxs)
        final_load_logicals = np.zeros(len(seq_idxs))
        for idx, img_num in enumerate(load_imgs):
            these_logicals, param_list,_ = self.get_logicals(img_idxs=img_num, site_idxs=load_site_idxs[idx],seq_idxs=seq_idxs)
            these_logicals = np.squeeze(np.all(these_logicals, axis=1))
            final_load_logicals = np.logical_and(final_load_logicals, these_logicals)
        final_surv_logicals = np.zeros(len(seq_idxs))
        for idx, img_num in enumerate(surv_imgs):
            these_logicals, param_list,_ = self.get_logicals(img_idxs=img_num, site_idxs=surv_site_idxs[idx],seq_idxs=seq_idxs)
            these_logicals = np.squeeze(np.all(these_logicals, axis=1))
            final_surv_logicals = np.logical_and(final_surv_logicals, these_logicals)
        return DataProcessTools.calculate_survival(final_load_logicals, final_surv_logicals, param_list)
    
    def plot_avg_img(self, img_idxs=None, site_idxs=np.array([])):
        # The default behavior is to plot no sites!
        img_idxs = utils.cvt_arg(img_idxs, self.num_imgs)
        n_imgs = len(img_idxs)
        if not isinstance(site_idxs, list):
            site_idxs = [utils.cvt_arg(site_idxs, self.num_sites) for i in range(n_imgs)]
        # site_idxs = utils.cvt_arg(site_idxs, self.num_sites)
        av_images = self.data['Analysis']['SummaryData']['av_images']
        imgs_to_plot = av_images[:,:,img_idxs - 1]
        sites_to_plot = []
        box_labels = []
        titles = []
        for idx in range(n_imgs):
            if len(site_idxs[idx]) != 0:
                these_sites = self.data['Scan']['SingleAtomSites'][img_idxs[idx] - 1][site_idxs[idx] - 1, :]
                box_labels.append([str(site_num) for site_num in site_idxs[idx]])
            else:
                these_sites = []
                box_labels.append([])
            sites_to_plot.append(these_sites)
            titles.append('Img #' + str(img_idxs[idx]) + ' ' + self.data['Scan']['SingleAtomSpecies'][img_idxs[idx] - 1])

        return PlotProcessTools.plot_images(imgs_to_plot, sites=sites_to_plot, box_size=self.data['Scan']['BoxSize'][()], box_label=box_labels, titles=titles)

    def plot_dual_imgs(self, img_idxs, site_idxs = np.array([]), titles=None):
        # The default behavior is to plot no sites!
        n_imgs = len(img_idxs)
        if not isinstance(site_idxs, list):
            site_idxs = [[utils.cvt_arg(site_idxs, self.num_sites), utils.cvt_arg(site_idxs, self.num_sites)] for i in range(n_imgs)]
        av_imgs = self.data['Analysis']['SummaryData']['av_images']
        imgs_to_plot = []
        sites_to_plot = []
        box_labels = []
        for idx in range(n_imgs):
            if isinstance(img_idxs[idx], list):
                img_idxs[idx] = np.array(img_idxs[idx])
            imgs_to_plot.append(av_imgs[:,:, img_idxs[idx] - 1])
            if len(site_idxs[idx][0]) != 0:
                if isinstance(site_idxs[idx][0], list):
                    site_idxs[idx][0] = np.array(site_idxs[idx][0])
                if isinstance(site_idxs[idx][1], list):
                    site_idxs[idx][1] = np.array(site_idxs[idx][1])
                these_sites = [self.data['Scan']['SingleAtomSites'][img_idxs[idx][0] - 1][site_idxs[idx][0] - 1, :], self.data['Scan']['SingleAtomSites'][img_idxs[idx][1] - 1][site_idxs[idx][1] - 1, :]]
                box_labels.append([[str(site_num) for site_num in site_idxs[idx][0]], [str(site_num) for site_num in site_idxs[idx][1]]])
            else:
                these_sites = [[],[]]
                box_labels.append([[], []])
            sites_to_plot.append(these_sites)
        return PlotProcessTools.plot_dual_images(imgs_to_plot, sites_to_plot, box_size = self.data['Scan']['BoxSize'][()], box_label=box_labels, titles=titles)
        # return PlotProcessTools.plot_dual_images(imgs_to_plot, sites_to_plot, box_size = 4, box_label=box_labels, titles=titles)

    def plot_avg_img_by_param(self, img_idxs=None, site_idxs=np.array([]), seq_idxs=None, titles=None):
        # Plot an average image by parameter. This would require loading in all images, but can be time-binned with seq_idxs
        # TODO: img_idxs should be a single element only
        # TODO: default titling
        mean_imgs = self.get_avg_img_by_param(img_idxs=img_idxs, seq_idxs=seq_idxs)
        n_imgs = mean_imgs.shape[3]
        if not isinstance(site_idxs, list):
            site_idxs = [utils.cvt_arg(site_idxs, self.num_sites) for i in range(n_imgs)]
        box_labels = []
        sites_to_plot = []
        for idx in range(n_imgs):
            if len(site_idxs[idx]) != 0:
                these_sites = self.data['Scan']['SingleAtomSites'][img_idxs[0] - 1][site_idxs[idx] - 1, :]
                box_labels.append([str(site_num) for site_num in site_idxs[idx]])
            else:
                these_sites = []
                box_labels.append([])
            sites_to_plot.append(these_sites)
        return PlotProcessTools.plot_images(np.squeeze(mean_imgs[:,:,0,:]), sites=sites_to_plot, box_size=self.data['Scan']['BoxSize'][()], box_label=box_labels, titles=titles)

    def plot_histograms(self, img_idxs=None, site_idxs=None, seq_idxs=None):
        # plot_histograms(sigs, plot_titles=None):
        
        img_idxs = utils.cvt_arg(img_idxs, self.num_imgs)
        seq_idxs = utils.cvt_arg(seq_idxs, self.num_seqs)
        site_idxs = utils.cvt_arg(site_idxs, self.num_sites)

        sigs,_,_ = self.get_signals(img_idxs=img_idxs, site_idxs =site_idxs, seq_idxs=seq_idxs)
        # Make titles list
        titles = []
        for idx in range(len(img_idxs)):
            this_title = []
            for site_idx in range(len(site_idxs)):
                this_title.append('Img ' + str(img_idxs[idx]) + ' Site ' + str(site_idxs[site_idx]))
            titles.append(this_title)
        return PlotProcessTools.plot_histograms(sigs, plot_titles=titles)
    
    def plot_histograms_by_param(self, img_idxs=None, site_idxs=None, seq_idxs=None):
        #TODO: Multiple imgs
        sigs = self.get_signals_by_param(img_idxs=img_idxs, site_idxs=site_idxs, seq_idxs=seq_idxs)
        # Make titles list
        titles = []
        for idx in range(len(sigs)): # len(sigs) is the number of parameters
            this_title = []
            for site_idx in range(len(site_idxs)):
                this_title.append('Param ' + str(idx) + ' Site ' + str(site_idxs[site_idx]))
            titles.append(this_title)
        return PlotProcessTools.plot_histograms(sigs, plot_titles=titles)

    def get_scan_param(self, scan_idx, dim, var_idx):
        # get a scan parameter for a scan_idx in the ScanGroup along dimension dim and variable var_idx
        # Note, variables are specified in alphabetical order.
        # Note, every index is 1 indexed
        scgrp = self.data['Scan']['ScanGroup']
        if isinstance(scgrp['scans']['baseidx'],list):
            nscans = int(len(scgrp['scans']['baseidx']))
        else:
            nscans = 1
        if scan_idx > nscans:
            raise Exception('scan_idx out of range. Total: ' + str(nscans) + ' scans.')
        this_scan = self._get_full_scan(scan_idx)
        this_vars_dict = this_scan['vars']['params'] # All the parameters for a given scan dimension
        if isinstance(this_vars_dict, dict):
            # Only a single dimension
            if dim > 1:
                raise Exception('Only a single dimension found for this scan.')
            else:
                key = list(this_vars_dict.keys())[var_idx - 1]
                return utils.obtain_recursive_key_and_value(this_vars_dict, "")
        else:
            this_vars_dict = this_vars_dict[dim - 1]
            key = list(this_vars_dict.keys())[var_idx - 1]
            return utils.obtain_recursive_key_and_value(this_vars_dict[key], key)
        
    def get_fixed_param(self, scan_idx):
        # Gets the fixed parameters for a given scan
        scgrp = self.data['Scan']['ScanGroup']
        if isinstance(scgrp['scans']['baseidx'],list):
            nscans = int(len(scgrp['scans']['baseidx']))
        else:
            nscans = 1
        if scan_idx > nscans:
            raise Exception('scan_idx out of range. Total: ' + str(nscans) + ' scans.')
        this_scan = self._get_full_scan(scan_idx)
        return this_scan['params']

    def _get_full_scan(self, scan_idx):
        # Gets the full scan for scan_idx in the ScanGroup. This merges the fixed parameters and the variable parameters
        # giving precedence to the non-base scan
        scgrp = self.data['Scan']['ScanGroup']
        if isinstance(scgrp['scans']['baseidx'],list):
            nscans = int(len(scgrp['scans']['baseidx']))
        else:
            nscans = 1
        if scan_idx > nscans:
            raise Exception('scan_idx out of range. Total: ' + str(nscans) + ' scans.')
        if nscans == 1:
            this_base_idx = scgrp['scans']['baseidx']
            this_params = scgrp['scans']['params']
            this_vars = scgrp['scans']['vars']
        else:
            this_base_idx = scgrp['scans']['baseidx'][scan_idx - 1]
            this_params = scgrp['scans']['params'][scan_idx - 1]
            this_vars = scgrp['scans']['vars'][scan_idx - 1]
        base_params = scgrp['base']['params']
        base_vars = scgrp['base']['vars']
        # Get rid of any repeated parameters from the base
        if isinstance(base_params, dict) and isinstance(this_params, dict):
            base_params = utils.remove_repeated_keys(base_params, this_params)
        if isinstance(base_vars, dict) and isinstance(this_vars, dict):
            for idx1, item1 in enumerate(base_vars['params']):
                for idx2, item2 in enumerate(this_vars['params']):
                    base_vars['params'][idx1] = utils.remove_repeated_keys(item1, item2)
        if isinstance(base_params, dict) and isinstance(this_vars, dict):
            for idx2, item2 in enumerate(this_vars['params']):
                base_params = utils.remove_repeated_keys(base_params, item2)
        if isinstance(base_vars, dict) and isinstance(this_params, dict):
            for idx1, item1 in enumerate(base_vars['params']):
                base_vars['params'][idx1] = utils.remove_repeated_keys(item1, this_params)

        # Merge base and this scan
        if isinstance(this_params, dict) and isinstance(base_params, dict):
            merged_params = utils.merge_dicts_with_lists(base_params, this_params)
        elif isinstance(this_params, dict):
            merged_params = this_params
        elif isinstance(base_params, dict):
            merged_params = base_params
        else:
            merged_params = base_params # Maintain the same weird convention for an empty MATLAB struct
        if isinstance(this_vars, dict) and isinstance(base_vars, dict):
            merged_vars = utils.merge_dicts_with_lists(base_vars, this_vars)
        elif isinstance(this_vars, dict):
            merged_vars = this_vars
        elif isinstance(base_vars, dict):
            merged_vars = base_vars
        else:
            merged_vars = base_vars # Maintain the same weird convention for an empty MATLAB struct
        return {'base_idx': this_base_idx, 'params': merged_params, 'vars': merged_vars}

    def _load_logicals(self, seq_idxs : np.ndarray):
        """
            Private method that loads logicals into the class

            Args:
                seq_idxs: the idxs of the sequences that need to be loaded. If sequence is already loaded, it will not be reloaded.
            Returns:
                None
            Raises:
                None

        """
        # Error checking
        if len(seq_idxs) == 0:
            seq_idxs = np.array([i + 1 for i in range(self.num_seqs)])
        if isinstance(seq_idxs, list):
            seq_idxs = np.array(seq_idxs)
        if seq_idxs.ndim != 1:
            print('Seq Idxs need to be 1 dimensional')
            return
        if any(seq_idxs <= 0):
            raise Exception('Seq Idxs are positive! (They are 1 indexed)')
        seq_idxs = np.unique(seq_idxs)
        seq_idxs = np.setdiff1d(seq_idxs, self.seq_idxs) # This ensures we only load new sequences
        # We define lists where we will fill in information before concatenating with what we already have. This ensures atomicity.
        nseqs = int(seq_idxs.shape[0])
        if nseqs == 0:
            return 
        temp_logicals = np.zeros([self.num_imgs, self.num_sites, nseqs])
        temp_signals = np.zeros([self.num_imgs, self.num_sites, nseqs])
        temp_params = np.zeros([nseqs])
        # First, we fill the cache if not already filled. The cache is only not filled upon the very first call.
        if self.log_files_cache is None:
            temp_log_files_cache = [0]
            for fname in self.summary_fnames:
                # construct the correct filenames
                dirname = os.path.split(self.master_fname)[0]
                full_fname = os.path.join(dirname, fname)
                # load the file
                try:
                    with h5py.File(full_fname) as f:
                    # Note, logicals are nimgs x nsites x nseqs
                        temp_log_files_cache.append(temp_log_files_cache[-1] + f['Analysis/SingleAtomLogical'][()].shape[0])
                        seq_idx_logicals = np.logical_and(seq_idxs > temp_log_files_cache[-2], seq_idxs <= temp_log_files_cache[-1]) # bool array of which seq_idxs are relevant here.
                        idxs_to_fill = seq_idxs.nonzero()[0] # find which ones are nonzero
                        if len(idxs_to_fill) != 0:
                            idxs_to_use = seq_idxs[seq_idx_logicals] - temp_log_files_cache[-2] - 1
                            sal_array = utils.cvt_multidim_array(f, 'Analysis/SingleAtomLogical')
                            sig_array = utils.cvt_multidim_array(f, 'Analysis/SummaryData/single_atom_signal')
                            temp_logicals[:, :, idxs_to_fill] = sal_array[:, :, idxs_to_use]
                            temp_signals[:,:, idxs_to_fill] = sig_array[:, :, idxs_to_use]
                            temp_params[idxs_to_fill] = np.squeeze(f['ParamList'][()][idxs_to_use,:])
                except FileNotFoundError:
                    print('File not found.')
                    return
                except Exception as e:
                    print('Unknown error when opening logical files')
                    return
            self.log_files_cache = temp_log_files_cache
            if any(seq_idxs > self.log_files_cache[-1]):
                raise Exception('Seq Idx out of range. Total: ' + str(self.log_files_cache[-1]) + ' sequences.')
        else:
            if any(seq_idxs > self.log_files_cache[-1]):
                raise Exception('Seq Idx out of range. Total: ' + str(self.log_files_cache[-1]) + ' sequences.')
            # Here, we assume the cache is already filled
            for idx in range(len(self.summary_fnames)):
                # Check if current file is relevant from the cache.
                seq_idx_logicals = np.logical_and(seq_idxs > self.log_files_cache[idx], seq_idxs <= self.log_files_cache[idx + 1]) # bool array of which seq_idxs are relevant here.
                idxs_to_fill = seq_idxs.nonzero()[0] # find which ones are nonzero
                if len(idxs_to_fill) != 0:
                    # construct the correct filenames
                    dirname = os.path.split(self.master_fname)[0]
                    full_fname = os.path.join(dirname, self.summary_fnames[idx])
                    # load the file
                    try:
                        with h5py.File(full_fname) as f:
                        # Note, logicals are nseqs x nsites x nimgs
                            idxs_to_use = seq_idxs[seq_idx_logicals] - self.log_files_cache[idx] - 1
                            sal_array = utils.cvt_multidim_array(f, 'Analysis/SingleAtomLogical')
                            sig_array = utils.cvt_multidim_array(f, 'Analysis/SummaryData/single_atom_signal')
                            temp_logicals[:, :, idxs_to_fill] = sal_array[:, :, idxs_to_use]
                            temp_signals[:,:, idxs_to_fill] = sig_array[:, :, idxs_to_use]
                            temp_params[idxs_to_fill] = np.squeeze(f['ParamList'][()][idxs_to_use,:])
                    except FileNotFoundError:
                        print('File not found.')
                        return
                    except:
                        print('Unknown error when opening logical files')
                        return
        if self.logicals is None:
            self.logicals = temp_logicals
            self.signals = temp_signals
            self.param_list = temp_params
            self.seq_idxs = seq_idxs
        else:
            self.logicals = np.concatenate((self.logicals, temp_logicals), axis = 2)
            self.signals = np.concatenate((self.signals, temp_signals), axis = 2)
            self.param_list = np.concatenate((self.param_list, temp_params), axis = 0)
            self.seq_idxs = np.concatenate((self.seq_idxs, seq_idxs), axis = 0)

    def _load_imgs(self, seq_idxs : np.ndarray):
        """
            Private method that loads images into the class

            Args:
                seq_idxs: the idxs of the sequences that need to be loaded
            Returns:
                None
            Raises:
                None

        """
        # Error checking
        # set_trace()
        if len(seq_idxs) == 0:
            seq_idxs = np.array([i + 1 for i in range(self.num_seqs)])
        if seq_idxs.ndim != 1:
            print('Seq Idxs need to be 1 dimensional')
            return
        if any(seq_idxs <= 0):
            raise Exception('Seq Idxs are positive! (They are 1 indexed)')
        seq_idxs = np.unique(seq_idxs)
        seq_idxs = np.setdiff1d(seq_idxs, self.seq_idxs_img) # This ensures we only load new sequences
        # We define lists where we will fill in information before concatenating with what we already have. This ensures atomicity.
        nseqs = int(seq_idxs.shape[0])
        if nseqs == 0:
            return
        frame_x = int(self.data['Scan']['FrameSize'][0])
        frame_y = int(self.data['Scan']['FrameSize'][1])
        nimgs_per_seq = int(self.data['Scan']['ImgsToSave'].size)
        temp_imgs = np.zeros([frame_x, frame_y, nimgs_per_seq, nseqs])
        # First, we fill the cache if not already filled. The cache is only not filled upon the very first call.
        if self.img_files_cache is None:
            temp_img_files_cache = [0]
            for idx, fname in enumerate(self.img_fnames):
                # construct the correct filenames
                print('Loading file ' + str(idx + 1) + ' out of ' + str(len(self.img_fnames)))
                dirname = os.path.split(self.master_fname)[0]
                full_fname = os.path.join(dirname, fname)
                # load the file
                # try:
                with h5py.File(full_fname) as f:
                # Note, logicals are nimgs x nsites x nseqs
                    temp_img_files_cache.append(temp_img_files_cache[-1] + f['Images'][()].shape[0] / nimgs_per_seq)
                    seq_idx_logicals = np.logical_and(seq_idxs > temp_img_files_cache[-2], seq_idxs <= temp_img_files_cache[-1]) # bool array of which seq_idxs are relevant here.
                    idxs_to_fill = seq_idx_logicals.nonzero()[0] # find which ones are nonzero
                    if len(idxs_to_fill) != 0:
                        idxs_to_use = seq_idxs[seq_idx_logicals] - temp_img_files_cache[-2] - 1
                        img_array = utils.cvt_multidim_array(f, 'Images')
                        # need to reshape
                        img_array = np.transpose(np.reshape(img_array, [frame_x, frame_y, len(idxs_to_fill), nimgs_per_seq]), (0,1,3,2))
                        # set_trace()
                        temp_imgs[:, :, :, idxs_to_fill.astype(np.int64)] = img_array[:, :, :, idxs_to_use.astype(np.int64)]
                # except FileNotFoundError:
                #     print('File not found.')
                #     return
                # except Exception as e:
                #     print('Unknown error when opening logical files')
                #     return
            self.img_files_cache = temp_img_files_cache
            if any(seq_idxs > self.img_files_cache[-1]):
                raise Exception('Seq Idx out of range. Total: ' + str(self.img_files_cache[-1]) + ' sequences.')
        else:
            if any(seq_idxs > self.img_files_cache[-1]):
                raise Exception('Seq Idx out of range. Total: ' + str(self.img_files_cache[-1]) + ' sequences.')
            # Here, we assume the cache is already filled
            for idx in range(self.img_fnames):
                # Check if current file is relevant from the cache.
                seq_idx_logicals = seq_idxs > self.img_files_cache[idx] & seq_idxs <= self.img_files_cache[idx + 1] # bool array of which seq_idxs are relevant here.
                idxs_to_fill = seq_idx_logicals.nonzero()[0] # find which ones are nonzero
                if len(idxs_to_fill) != 0:
                    # construct the correct filenames
                    dirname = os.path.split(self.master_fname)[0]
                    full_fname = os.path.join(dirname, self.img_fnames[idx])
                    # load the file
                    # try:
                    with h5py.File(full_fname) as f:
                    # Note, logicals are nseqs x nsites x nimgs
                        idxs_to_use = seq_idxs[seq_idx_logicals] - self.img_files_cache[idx] - 1
                        img_array = utils.cvt_multidim_array(f, 'Images')
                        # need to reshape
                        img_array = np.transpose(np.reshape(img_array, [frame_x, frame_y, len(idxs_to_fill), nimgs_per_seq]), (0,1,3,2))
                        temp_imgs[:, :, :, idxs_to_fill] = img_array[:, :, :, idxs_to_use]
                    # except FileNotFoundError:
                    #     print('File not found.')
                    #     return
                    # except:
                    #     print('Unknown error when opening logical files')
                    #     return
        if self.imgs is None:
            self.imgs = temp_imgs
            self.seq_idxs_img = seq_idxs
        else:
            self.imgs = np.concatenate((self.imgs, temp_imgs), axis = 3)
            self.seq_idxs_img = np.concatenate((self.seq_idxs_img, seq_idxs), axis = 0)