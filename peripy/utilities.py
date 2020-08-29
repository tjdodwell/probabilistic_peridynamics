"""Utility functions that are unrelated to peridynamics."""
import h5py
import warnings


def write_array(write_path, dataset, array):
    """
    Write a :class: numpy.ndarray to a HDF5 file.

    :arg write_path: The path to which the HDF5 file is written.
    :type write_path: path-like or str
    :arg dataset: The name of the dataset stored in the HDF5 file.
    :type dataset: str
    :array: The array to be written to file.
    :type array: :class: numpy.ndarray

    :return: None
    :rtype: None type
    """
    with h5py.File(write_path, 'a') as hf:
        hf.create_dataset(dataset,  data=array)


def read_array(read_path, dataset):
    """
    Read a :class numpy.ndarray: from a HDF5 file.

    :arg read_path: The path to which the HDF5 file is written.
    :type read_path: path-like or str
    :arg dataset: The name of the dataset stored in the HDF5 file.
    :type dataset: str

    :return: An array which was stored on disk.
    :rtype: :class numpy.ndarray:
    """
    try:
        with h5py.File(read_path, 'r') as hf:
            try:
                array = hf[dataset][:]
                return array
            except KeyError:
                warnings.warn(
                    "The {} array does not appear to exist in the file {}. "
                    "Please set a write_path keyword argument in `Model` "
                    "and the {} array will be created and then written to "
                    "that file path.".format(dataset, read_path, dataset))
    except IOError:
        warnings.warn(
            "The {} file does not appear to exist yet.".format(
                read_path))
        return None
