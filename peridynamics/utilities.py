"""Utilities that are unrelated to peridynamics."""
import h5py
import numpy as np


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
    
def _calc_midpoint_gradient(T, displacement_scale_rate):
    """
    Calculate the midpoint gradient and coefficients of a 5th order polynomial.

    Calculates the midpoint gradient and coefficients of a 5th order
    polynomial displacement-time curve which is defined by acceleration being
    0 at t=0 and t=T and a midpoint gradient.

    :arg int T: The finish time-step of the displacement-time curve.
    :arg float displacement_scale_rate: The midpoint gradient of the curve.

    :returns: A tuple containing a float
        the midpoint gradient of the displacement-time curve and a tuple
        containing the 3 unconstrained coefficients of the 5th-order
        polynomial.
    :rtype: A tuple containing (:type float:, :type tuple:)
    """
    A = np.array([
        [(1 * T**5) / 1, (1 * T**4) / 1, (1 * T**3) / 1],
        [(20 * T**3) / 1, (12 * T**2) / 1, (6 * T**1) / 1],
        [(5 * T**4) / 1, (4 * T**3) / 1, (3 * T**2) / 1]
        ]
        )
    b = np.array(
        [
            [displacement_scale_rate],
            [0.0],
            [0.0]
                ])
    x = np.linalg.solve(A, b)
    a = x[0][0]
    b = x[1][0]
    c = x[2][0]
    midpoint_gradient = (5./16)*a*T**4 + (4./8)*b*T**3 + (3./4)*c*T**2
    coefficients = (a, b, c)
    return(midpoint_gradient, coefficients)


def calc_displacement_scale(
        coefficients, final_displacement, build_time, displacement_rate, step,
        build_displacement, ease_off):
    """
    Calculate the displacement scale.

    Calculates the displacement boundary condition scale according to a
    5th order polynomial/ linear displacement-time curve for which initial
    acceleration is 0.

    :arg tuple coefficients: Tuple containing the 3 free coefficients
        of the 5th order polynomial.
    :arg float final_displacement: The final applied displacement in [m].
    :arg int build_time: The number of time steps over which the
        applied displacement-time curve is not linear.
    :arg float displacement_rate: The displacement rate in [m] per step
        during the linear phase of the displacement-time graph.
    :arg int step: The current time-step of the simulation.
    :arg float build_displacement: The displacement in [m] over which the
        displacement-time graph is the smooth 5th order polynomial.
    :arg int ease_off: A boolean-like variable which is 0 if the
        displacement-rate hasn't started decreasing yet. Equal to the step
        at which the displacement rate starts decreasing once it does so.

    :returns: The displacement_scale between [0.0, 1.0], a scale
        applied to the displacement boundary conditions.
    :rtype: np.float64
    """
    a, b, c = coefficients
    if step < build_time / 2:
        m = 5 * a * step**4 + 4 * b * step**3 + 3 * c * step**2
        displacement_scale = m / displacement_rate
    elif ease_off != 0:
        t = step - ease_off + build_time / 2
        if t > build_time:
            displacement_scale = 0.0
        else:
            m = 5 * a * t**4 + 4 * b * t**3 + 3 * c * t**2
            displacement_scale = m / displacement_rate
    else:  # linear increments
        # calculate displacement
        linear_time = step - build_time/2
        linear_displacement = linear_time * displacement_rate
        displacement = linear_displacement + build_displacement/2
        if displacement + build_displacement / 2 < final_displacement:
            displacement_scale = 1.0
        else:
            ease_off = step
            displacement_scale = 1.0
    return(displacement_scale, ease_off)


def calc_build_time(build_displacement, displacement_rate, steps):
    """
    Calculate the the number of steps for the 5th order polynomial.

    An iterative procedure to calculate the number of steps over which the
    displacement-time curve is a smooth 5th order polynomial.

    :arg float build_displacement: The displacement in [m] over which the
        displacement-time graph is the smooth 5th order polynomial.
    :arg float displacement_rate: The displacement rate in [m] per step
            during the linear phase of the displacement-time graph.
    :arg int step: The current time-step of the simulation.

    :returns: A tuple containing an int T the number of steps over which the
        displacement-time curve is a smooth 5th order polynomial and a tuple
        containing the 3 unconstrained coefficients of the 5th-order
        polynomial.
    :rtype: A tuple containing (:type int:, :type tuple:)
    """
    build_time = 0
    midpoint_gradient = np.inf
    while midpoint_gradient > displacement_rate:
        # Try to calculate gradient
        try:
            midpoint_gradient, coefficients = _calc_midpoint_gradient(
                build_time, build_displacement)
        # No solution.. try increasing the build_time
        except Exception:
            pass
        build_time += 1
        if build_time > steps:
            raise ValueError(
                'Displacement build-up time was larger than total simulation \
                time steps! \ntry decreasing build_displacement, or\
                increasing max_displacement_rate. steps = {}'.format(steps))
            break
    return(build_time, coefficients)