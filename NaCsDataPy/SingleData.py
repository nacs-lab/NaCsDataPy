from ast import Load
import os
import h5py
import numpy as np

import utils

from enum import Enum

class LoadStatus(Enum):
    NoneLoaded = 0 # This is O(1) data, like averages
    LogicalsLoaded = 1 # In particular, this is all the O(n) data except images
    ImgLoaded = 2 # These are images
    AllLoaded = 3 # Both images and logicals



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
        self.load_status : LoadStatus = LoadStatus.NoneLoaded
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
        self.locked = False # Once any modification is made, then the loader will be locked to reduce confusion. 

    def __del__(self):
        """ 
            Function for when the object is deleted. We need to make sure the file is closed.
        """
        self.data_file.close()
        self.names_file.close()

    def setLoadStatus(self, level: LoadStatus):
        """
            Function to set the LoadStatus of this object. Will reset the corresponding logicals and imgs arrays if the LoadStatus changes to one that no longer requires it

            Args:
                level: A LoadStatus.
            Returns:
                None
            Raises:
                None

        """
        if level is LoadStatus.NoneLoaded:
            self.seq_idxs = None
            self.logicals = None
            self.imgs = None
            self.load_status = LoadStatus.NoneLoaded
        elif level is LoadStatus.LogicalsLoaded:
            self.imgs = None
            self.load_status = LoadStatus.LogicalsLoaded
        elif level is LoadStatus.ImgLoaded:
            self.logicals = None
            self.load_status = LoadStatus.ImgLoaded
        elif level is LoadStatus.AllLoaded:
            self.load_status = LoadStatus.AllLoaded

    def get_logicals(self, seq_idxs=[], img_idxs=[]):
        # seq_idxs are sequence indices, 1 indexed
        # img_idxs are the images that are wanted. A negative number indicates that you want to apply the logical NOT.
        self._load_logicals(seq_idxs)


        if len(seq_idxs) == 0 and len(img_idxs) == 0:
            return self.logicals, self.param_list, self.seq_idxs
        if len(seq_idxs) == 0:
            abs_img_idxs = np.abs(img_idxs) - 1
            ret_logs = self.logicals[abs_img_idxs,:,:]
            if any(img_idxs < 0):
                mod_arr = ret_logs[np.where(img_idxs < 0),:,:]
                ret_logs[np.where(img_idxs < 0),:,:] = np.where(mod_arr == 0, 1, 0)
            return ret_logs, self.param_list, self.seq_idxs
        if isinstance(seq_idxs, list):
            seq_idxs = np.array(seq_idxs)
        idxs = np.where(np.in1d(seq_idxs, self.seq_idxs))[0] # np.where returns a tuple
        if len(img_idxs) == 0:
            return self.logicals[:,:,idxs], self.param_list[idxs], self.seq_idxs[idxs]
        else:
            abs_img_idxs = np.abs(img_idxs) - 1
            ret_logs = self.logicals[abs_img_idxs,:,idxs]
            if any(img_idxs < 0):
                mod_arr = ret_logs[abs_img_idxs[img_idxs < 0],:,:]
                ret_logs[abs_img_idxs[img_idxs < 0],:,:] = np.where(mod_arr == 0, 1, 0)
            return ret_logs[:,:, idxs], self.param_list[idxs], self.seq_idxs[idxs]
        
    def get_signals(self, seq_idxs=[], img_idxs=[]):
        # seq_idxs are sequence indices, 1 indexed
        # img_idxs are the images that are wanted.
        self._load_logicals(seq_idxs) # this will load the signals if they are not loaded

        if len(seq_idxs) == 0 and len(img_idxs) == 0:
            return self.signals, self.param_list, self.seq_idxs
        if len(seq_idxs) == 0:
            abs_img_idxs = np.abs(img_idxs) - 1
            ret_signals = self.signals[abs_img_idxs,:,:]
            return ret_signals, self.param_list, self.seq_idxs
        idxs = np.where(np.in1d(seq_idxs, self.seq_idxs))[0]
        if len(img_idxs) == 0:
            return self.signals[:,:,idxs], self.param_list[idxs], self.seq_idxs[idxs]
        else:
            abs_img_idxs = np.abs(img_idxs) - 1
            ret_signals = self.signals[abs_img_idxs,:,idxs]
            return ret_signals[:,:, idxs], self.param_list[idxs], self.seq_idxs[idxs]

    def load(self, seq_idxs=-1):
        """
            Loads in images or logicals depending on the current LoadStatus

            Args:
                seq_idxs:  the idxs of the sequences that need to be loaded (1 indexed)
            Returns:
                None
            Raises:
                None

        """
        if seq_idxs == -1:
            seq_idxs = np.array([i + 1 for i in range(self.num_seqs)])
        if self.load_status == LoadStatus.LogicalsLoaded or self.load_status == LoadStatus.AllLoaded:
            print('Loading Logicals')
            self._load_logicals(seq_idxs)
        if self.load_status == LoadStatus.ImgLoaded or self.load_status == LoadStatus.AllLoaded:
            print('Loading Images')
            self._load_imgs(seq_idxs)
    
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
        else:
            # Here, we assume the cache is already filled
            for idx in range(self.summary_fnames):
                # Check if current file is relevant from the cache.
                seq_idx_logicals = seq_idxs > self.log_files_cache[idx] & seq_idxs <= self.log_files_cache[idx + 1] # bool array of which seq_idxs are relevant here.
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
        if len(seq_idxs) == 0:
            seq_idxs = np.array([i + 1 for i in range(self.num_seqs)])
        if seq_idxs.ndim != 1:
            print('Seq Idxs need to be 1 dimensional')
            return
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
            for fname in self.img_fnames:
                # construct the correct filenames
                dirname = os.path.split(self.master_fname)[0]
                full_fname = os.path.join(dirname, fname)
                # load the file
                try:
                    with h5py.File(full_fname) as f:
                    # Note, logicals are nimgs x nsites x nseqs
                        temp_img_files_cache.append(temp_img_files_cache[-1] + f['Images'][()].shape[2] / nimgs_per_seq)
                        seq_idx_logicals = np.logical_and(seq_idxs > temp_img_files_cache[-2], seq_idxs <= temp_img_files_cache[-1]) # bool array of which seq_idxs are relevant here.
                        idxs_to_fill = seq_idxs.nonzero()[0] # find which ones are nonzero
                        if len(idxs_to_fill) != 0:
                            idxs_to_use = seq_idxs[seq_idx_logicals] - temp_img_files_cache[-2] - 1
                            img_array = utils.cvt_multidim_array(f, 'Images')
                            # need to reshape
                            img_array = np.transpose(np.reshape(img_array, [frame_x, frame_y, nseqs, nimgs_per_seq]), (0,1,3,2))
                            temp_imgs[:, :, :, idxs_to_fill] = img_array[:, :, :, idxs_to_use]
                except FileNotFoundError:
                    print('File not found.')
                    return
                except Exception as e:
                    print('Unknown error when opening logical files')
                    return
        else:
            # Here, we assume the cache is already filled
            for idx in range(self.img_fnames):
                # Check if current file is relevant from the cache.
                seq_idx_logicals = seq_idxs > self.img_files_cache[idx] & seq_idxs <= self.img_files_cache[idx + 1] # bool array of which seq_idxs are relevant here.
                idxs_to_fill = seq_idxs.nonzero()[0] # find which ones are nonzero
                if len(idxs_to_fill) != 0:
                    # construct the correct filenames
                    dirname = os.path.split(self.master_fname)[0]
                    full_fname = os.path.join(dirname, self.img_fnames[idx])
                    # load the file
                    try:
                        with h5py.File(full_fname) as f:
                        # Note, logicals are nseqs x nsites x nimgs
                            idxs_to_use = seq_idxs[seq_idx_logicals] - self.img_files_cache[idx] - 1
                            img_array = utils.cvt_multidim_array(f, 'Images')
                            # need to reshape
                            img_array = np.transpose(np.reshape(img_array, [frame_x, frame_y, nseqs, nimgs_per_seq]), (0,1,3,2))
                            temp_imgs[:, :, :, idxs_to_fill] = img_array[:, :, :, idxs_to_use]
                    except FileNotFoundError:
                        print('File not found.')
                        return
                    except:
                        print('Unknown error when opening logical files')
                        return
        if self.imgs is None:
            self.imgs = temp_imgs
        else:
            self.imgs = np.concatenate((self.imgs, temp_logicals), axis = 3)
            self.seq_idxs_img = np.concatenate((self.seq_idxs_img, seq_idxs), axis = 0)