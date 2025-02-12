import h5py
import numpy as np

def cvt_to_str(file: h5py._hl.files.File, fieldname: str):
    """
        Converts a cell array field of strings in matlab into a list of strings in python.

        It seems like it is stored as a reference, and then we need to tap into that reference, and convert from numbers to ASCII characters

        Args:
            file: The file returned by h5py.File which represents the file
            fieldname: The fieldname that needs to be accessed in the form of a string
        Returns:
            A list of strings corresponding to the field name.
        Raises:
            None
    """
    ref_arr : np.ndarray = file[fieldname][()]
    n_elems : int = len(ref_arr)
    res = []
    for idx in range(n_elems):
        # First, we obtain the reference location
        ref_name : str = h5py.h5r.get_name(ref_arr[idx][0], file.id).decode()
        str_arr : np.ndarray = file[ref_name][()]
        name_str : str = "".join([chr(int(item)) for item in str_arr]) # First two characters is .\
        res.append(name_str[2:])
    return res

def cvt_unicode_to_str(arr):
    return  "".join([chr(int(item)) for item in arr])

def cvt_multidim_array(file: h5py._hl.files.File, fieldname: str):
    """
        Converts a multidimensional array, stored from MATLAB in column-major order to the row-major order of numpy.

        Args:
            file: The file returned by h5py.File which represents the file
            fieldname: The fieldname that needs to be accessed in the form of a string
        Returns:
            A numpy array corresponding to the field.
        Raises:
            None
    """
    arr = file[fieldname][()]
    ndims = arr.ndim
    dim_array = []
    arr = np.transpose(arr, axes=np.flip([i for i in range(ndims)]))
    return arr

def cvt_to_dict(file: h5py._hl.files.File, fieldname :str  = ""):
    """Recursively convert an HDF5 group into a Python dictionary."""
    if fieldname == "":
        hdf5_group = file
    else:
        hdf5_group = file[fieldname]
    result = {}
    for key, item in hdf5_group.items():
        # skip '#refs#'
        if key == '#refs#':
            continue
        if isinstance(item, h5py.Dataset):  # If it's a dataset, extract the data
            temp_result = item[()]  # Convert dataset to NumPy array or scalar
            if isinstance(temp_result, np.ndarray):
                ndims = temp_result.ndim
                temp_result = np.transpose(temp_result, axes=np.flip([i for i in range(ndims)]))
                temp_result = np.squeeze(temp_result)
                # Convert any cell arrays to lists
                if isinstance(temp_result.flat[0], h5py.h5r.Reference):
                    temp_result = temp_result.tolist()
                    if isinstance(temp_result, list):
                        modify_list(temp_result, cvt_ref_to_data, file)
                    else:
                        temp_result = cvt_ref_to_data(temp_result, file)
            result[key] = temp_result
        elif isinstance(item, h5py.Group):  # If it's a group, recurse into it
            new_fieldname = fieldname + '/' + key
            result[key] = cvt_to_dict(file, new_fieldname)
    return result

def modify_list(nested_list, func, *args):
    """Recursively modifies a nested list by applying func to each element."""
    for i in range(len(nested_list)):
        if isinstance(nested_list[i], list):  # If element is a list, go deeper
            modify_list(nested_list[i], func)
        else:
            nested_list[i] = func(nested_list[i], *args)  # Modify in place

def cvt_ref_to_data(ref: h5py.h5r.Reference, file: h5py._hl.files.File):
    """Converts a reference to the actual data it points to."""
    ref_name : str = h5py.h5r.get_name(ref, file.id).decode()
    result = file[ref_name]
    if isinstance(result, h5py.Dataset):
        result = result[()]
        if isinstance(result, np.ndarray):
            ndims = result.ndim
            result = np.transpose(result, axes=np.flip([i for i in range(ndims)]))
            result = np.squeeze(result)
            # Convert any cell arrays to lists
            if isinstance(result.flat[0], h5py.h5r.Reference):
                result = result.tolist()
                if isinstance(result, list):
                    modify_list(result, cvt_ref_to_data, file)
                else:
                    result = cvt_ref_to_data(result, file)
    elif isinstance(result, h5py.Group):
        result = cvt_to_dict(file, ref_name)
    return result

def get_scangroup(file: h5py._hl.files.File):
    """
        TODO

        Args:
            TODO
        Returns:
            TODO
        Raises:
            None
    """
    # The six fields we need to extract are 'base', 'runparam', 'scans', 'use_var_base', 'use_var_scans', 'version'
    # The 'base' has fields 'params' and 'vars', indicating the fixed parameters and scannable parameters respectively of the base of the ScanGroup
    # The 'runparam' is the special structure that contains the run parameters
    # The 'scans' is an array of structs containing the fields 'params' and 'vars', indicating the fixed parameters and scannable parameters specific to each scan
    # We will ignore 'use_var_base' and 'use_var_scans' for now
    # The 'version' is the version of the ScanGroup.

    scgrp_out = dict()

    # The easy one is runparam
    scgrp_out.runparam = cvt_to_dict(file, "Scan/ScanGroup/runparam")

