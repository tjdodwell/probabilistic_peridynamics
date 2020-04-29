import pyopencl as cl

DOUBLE_FP_SUPPORT = (
    cl.device_fp_config.DENORM | cl.device_fp_config.FMA |
    cl.device_fp_config.INF_NAN | cl.device_fp_config.ROUND_TO_INF |
    cl.device_fp_config.ROUND_TO_NEAREST |
    cl.device_fp_config.ROUND_TO_ZERO
    )

for platform in cl.get_platforms():
    for device_type in [cl.device_type.GPU, cl.device_type.ALL]:
        print(cl.device_type.to_string(device_type))
        for device in platform.get_devices(device_type):
            print(f"\tDevice name: {device.get_info(cl.device_info.NAME)}")
            print(f"\tfp64 support: {device.get_info(cl.device_info.DOUBLE_FP_CONFIG) & DOUBLE_FP_SUPPORT != 0}\n")
