"""Utilities for using the OpenCL kernels."""
import pyopencl as cl
import sys


DOUBLE_FP_SUPPORT = (
    cl.device_fp_config.DENORM | cl.device_fp_config.FMA |
    cl.device_fp_config.INF_NAN | cl.device_fp_config.ROUND_TO_INF |
    cl.device_fp_config.ROUND_TO_NEAREST |
    cl.device_fp_config.ROUND_TO_ZERO
    )


def double_fp_support(device):
    """
    Test whether a context supports double floating-point precission.

    :arg device: The OpenCL context to test.
    :type device: :class:`pyopencl._cl.Device`

    :returns: `True` if the device supports double floating-point precision,
    `False` otherwise.
    :rtype: `bool`
    """
    return device.get_info(cl.device_info.DOUBLE_FP_CONFIG) & DOUBLE_FP_SUPPORT


def get_context():
    """
    Find an appropriate OpenCL context.

    This function looks for a device with support for double
    floating-point precision and prefers GPU devices.

    :returns: A context with a single suitable device, or `None` is no suitable
    device is found.
    :rtype: :class:`pyopencl._cl.Context` or `NoneType`
    """
    for platform in cl.get_platforms():
        for device_type in [cl.device_type.GPU, cl.device_type.ALL]:
            for device in platform.get_devices(device_type):
                if double_fp_support(device):
                    return cl.Context([device])
    return None


def output_device_info(device_id):
    """Output the device info of the device."""
    sys.stdout.write("Device is ")
    sys.stdout.write(device_id.name)
    if device_id.type == cl.device_type.GPU:
        sys.stdout.write("GPU from ")
    elif device_id.type == cl.device_type.CPU:
        sys.stdout.write("CPU from ")
    else:
        sys.stdout.write("non CPU of GPU processor from ")
    sys.stdout.write(device_id.vendor)
    sys.stdout.write(" with a max of ")
    sys.stdout.write(str(device_id.max_compute_units))
    sys.stdout.write(" compute units, \n")
    sys.stdout.write("a max of ")
    sys.stdout.write(str(device_id.max_work_group_size))
    sys.stdout.write(" work-items per work-group, \n")
    sys.stdout.write("a max work item dimensions of ")
    sys.stdout.write(str(device_id.max_work_item_dimensions))
    sys.stdout.write(", \na max work item sizes of ")
    sys.stdout.write(str(device_id.max_work_item_sizes))
    sys.stdout.write(",\nand device local memory size is ")
    sys.stdout.write(str(device_id.local_mem_size))
    sys.stdout.write(" bytes. \n")
    sys.stdout.flush()
    return 1
