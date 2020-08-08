"""Utilitie functions that are unrelated to peridynamics."""
import h5py
import numpy as np
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


def _calc_midpoint_gradient(T, displacement):
    """
    Calculate the midpoint gradient and coefficients of a 5th order polynomial.

    Calculates the midpoint gradient and coefficients of a 5th order
    polynomial displacement-time curve which is defined by acceleration being
    0 at t=0 and t=T and a total displacement.

    :arg int T: The total time in number of time steps of the smooth 5th order
        polynomial.
    :arg float displacement: The final displacement in [m] of the smooth 5th
        order polynomial.

    :returns: A tuple containing the midpoint gradient of the
        displacement-time curve and a tuple containing the 3 unconstrained
        coefficients of the 5th-order polynomial.
    :rtype: A tuple containing (:type float:, :type tuple:)
    """
    A = np.array([
        [1 * T**5, 1 * T**4, 1 * T**3],
        [20 * T**3, 12 * T**2, 6 * T],
        [5 * T**4, 4 * T**3, 3 * T**2]
        ]
        )
    b = np.array(
        [
            [displacement],
            [0.0],
            [0.0]
                ])
    x = np.linalg.solve(A, b)
    a = x[0][0]
    b = x[1][0]
    c = x[2][0]
    midpoint_gradient = (
        5 * a * (T / 2)**4 + 4 * b * (T/2)**3 + 3 * c * (T/2)**2)
    coefficients = (a, b, c)
    return(midpoint_gradient, coefficients)


def calc_displacement_magnitude(
        coefficients, max_displacement, build_time, max_displacement_rate,
        step, build_displacement, ease_off):
    """
    Calculate the displacement scale.

    Calculates the displacement boundary condition magnitude according to a
    5th order polynomial/ linear displacement-time curve for which initial
    acceleration is 0.

    :arg tuple coefficients: Tuple containing the 3 free coefficients
        of the 5th order polynomial.
    :arg float max_displacement: The final applied displacement in [m].
    :arg int build_time: The number of time steps over which the
        applied displacement-time curve is not linear.
    :arg float max_displacement_rate: The maximum displacement rate
        in [m] per step, which is the displacement rate during the linear phase
        of the displacement-time graph.
    :arg int step: The current time-step of the simulation.
    :arg float build_displacement: The displacement in [m] over which the
        displacement-time graph is the smooth 5th order polynomial.
    :arg int ease_off: A boolean-like variable which is 0 if the
        displacement-rate hasn't started decreasing yet. Equal to the step
        at which the displacement rate starts decreasing once it does so.

    :returns: The displacement_bc_magnitude between
        [0.0, max_displacement], a scale applied to the displacement
        boundary conditions.
    :rtype: np.float64
    """
    a, b, c = coefficients
    # Acceleration part of displacement-time curve.
    if step < build_time / 2:
        displacement_bc_magnitude = a * step**5 + b * step**4 + c * step**3
    # Deceleration part of dispalcement-time curve.
    elif ease_off != 0:
        t = step - ease_off + build_time / 2
        if t > build_time:
            displacement_bc_magnitude = max_displacement
        else:
            displacement_bc_magnitude = (
                a * t**5 + b * t**4 + c * t**3
                + max_displacement
                - build_displacement)
    # Constant velocity
    else:
        # Calculate displacement.
        linear_time = step - build_time / 2
        linear_displacement = linear_time * max_displacement_rate
        displacement = linear_displacement + build_displacement / 2
        if displacement + build_displacement / 2 < max_displacement:
            displacement_bc_magnitude = displacement
        else:
            ease_off = step
            displacement_bc_magnitude = (
                max_displacement - build_displacement / 2)
    return(displacement_bc_magnitude, ease_off)


def calc_build_time(build_displacement, max_displacement_rate, steps):
    """
    Calculate the the number of steps for the 5th order polynomial.

    An iterative procedure to calculate the number of steps over which the
    displacement-time curve is a smooth 5th order polynomial.

    :arg float build_displacement: The displacement in [m] over which the
        displacement-time graph is the smooth 5th order polynomial.
    :arg float max_displacement_rate: The displacement rate in [m] per step
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
    while midpoint_gradient > max_displacement_rate:
        # Try to calculate gradient
        try:
            midpoint_gradient, coefficients = _calc_midpoint_gradient(
                build_time, build_displacement)
        # No solution, so increase the build_time
        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                pass
        build_time += 1
        if build_time > steps:
            raise ValueError(
                "Displacement build-up time was larger than total simulation "
                "time steps! \nTry increasing steps, decreasing "
                "build_displacement, or increasing max_displacement_rate. "
                "steps = {}".format(steps))
            break
    return(build_time, coefficients)


def calc_boundary_conditions_magnitudes(
        steps, max_displacement_rate, max_displacement=None,
        build_displacement=0, build_load_steps=None, max_load=None):
    """
    Caclulate the boundary condition magnitudes arrays.

    Calculates the magnitude applied to the boundary conditions for each time-
    step for displacement and force boundary conditions.

    :arg float max_displacement_rate: The displacement rate in [m] per step
        during the linear phase of the displacement-time graph, and the
        maximum displacement rate of any part of the simulation.
    :arg int steps: The number of simulation steps to conduct.
    :arg float max_displacement: The final applied displacement in [m].
        Default is 0.
    :arg float build_displacement: The displacement in [m] over which the
        displacement-time graph is the smooth 5th order polynomial.
    :arg float build_load_steps: The inverse of the number of steps
        required to build up to full external force loading.
    :arg float max_load: The maximum total external load in [N] applied to the
        loaded nodes.

    :returns tuple: A tuple of the (steps) displacement_bc_array and the
        (steps) force_bc_array which are the magnitudes applied to the
        displacement and force boundary conditions over each step,
        respectively.
    :rtype tuple (numpy.ndarray, numpy.ndarray):
    """
    # Calculate no. of time steps that applied BCs are in the build phase
    if not ((max_displacement_rate is None)
            or (build_displacement is None)
            or (max_displacement is None)):
        build_time, coefficients = calc_build_time(
            build_displacement, max_displacement_rate, steps)
    else:
        build_time, coefficients = None, None

    displacement_bc_array = []
    force_bc_array = []

    ease_off = 0
    for step in range(steps):
        # Increase external forces in linear incremenets
        force_bc_array.append(
            _increment_load(build_load_steps, max_load, step))

        # Increase displacement in 5th order polynomial increments
        displacement_bc_magnitude, ease_off = _increment_displacement(
            coefficients, build_time, step, ease_off,
            max_displacement_rate, build_displacement, max_displacement)
        displacement_bc_array.append(
            displacement_bc_magnitude)

    return (np.array(displacement_bc_array), np.array(force_bc_array))
