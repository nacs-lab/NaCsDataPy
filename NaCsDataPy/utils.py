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

def merge_dicts_with_lists(dict1, dict2):
    merged = dict1.copy()  # Create a copy of dict1 to avoid modifying it
    for key, value in dict2.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dicts_with_lists(merged[key], value)  # Recursively merge
        elif key in merged and isinstance(merged[key], dict) and isinstance(value, list):
            continue
        elif key in merged and isinstance(merged[key], list) and isinstance(value, dict):
            merged[key] = value # Always take the dict
        elif key in merged and isinstance(merged[key], list) and isinstance(value, list):
            merged[key] = merge_lists_with_dicts(merged[key], value)
        else:
            merged[key] = value  # Overwrite with dict2's value
    return merged

def merge_lists_with_dicts(list1, list2):
    final_list = list1.copy()
    max_dict1 = len(list1)
    max_dict2 = len(list2)
    for i in range(max(max_dict1, max_dict2)):
        if i >= max_dict1:
            final_list[i] = list2[i]
        elif i >= max_dict2:
            continue # Don't do anything
        else:
            if isinstance(final_list[i], dict) and isinstance(list2[i], dict):
                final_list[i] = merge_dicts_with_lists(final_list[i], list2[i])
            elif isinstance(final_list[i], dict) and isinstance(list2[i], list):
                # Keep the dict if there is one
                continue
            elif isinstance(final_list[i], list) and isinstance(list2[i], dict):
                final_list[i] = list2[i]
            elif isinstance(final_list[i], list) and isinstance(list2[i], list):
                final_list[i] = merge_lists_with_dicts(final_list[i], list2[i])
            else:
                final_list[i] = list2[i]
    return final_list

def obtain_recursive_key_and_value(val, key_str = ""):
    # This function assumes that it can be a nested dictionary, but only one key at each step.
    # It returns the final value, and the key path to reach that value in string format.
    if isinstance(val, dict):
        keys = list(val.keys())
        if len(keys) > 1:
            print("Dictionary has more than one key at this level. Taking first key.")
        sub_key = keys[0]
        if key_str == "":
            new_key_str = sub_key
        else:
            new_key_str = key_str + "." + sub_key
        sub_val, sub_path = obtain_recursive_key_and_value(val[sub_key], new_key_str)
        return sub_val, sub_path
    else:
        return val, key_str