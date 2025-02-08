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
        name_str : str = "".join([chr(item) for item in str_arr]) # First two characters is .\
        res.append(name_str[2:])
    return res

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
