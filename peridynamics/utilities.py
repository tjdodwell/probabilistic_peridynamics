"""Utilities for reading network.h5 files."""
import h5py


def read_array(read_path, dataset):
    """
    Read a :class numpy.ndarray: from a HDF5 file.

    :arg write_path: The path to which the HDF5 file is written.
    :type write_path: path-like or str
    :arg dataset: The name of the dataset stored in the HDF5 file.
    :type dataset: str

    :return: array
    :rtype: :class numpy.ndarray:
    """
    try:
        with h5py.File(read_path, 'r') as hf:
            array = hf[dataset][:]
        return array
    except IOError:
        
        print("The .h5 file at {} does not appear to exist, the {} array will"
              " be created and then written to file instead".format(
              read_path, dataset))
        return None