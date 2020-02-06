class Euler(object):
    r"""
    Euler integrator.

    The Euler method is a first-order numerical integration method. The
    integration is given by,

    .. math::
        u(t + \delta t) = u(t) + \delta t f(t) d

    where :math:`u(t)` is the displacement at time :math:`t`, :math:`f(t)` is
    the force at time :math:`t`, :math:`\delta t` is the time step and
    :math:`d` is a dampening factor.
    """
    def __init__(self, dt, dampening=1.0):
        """
        Create a :class:`Euler` integrator object.

        :arg float dt: The integration time step.
        :arg float dampening: The dampening factor. The default is 1.0

        :returns: A :class:`Euler` object
        """
        self.df = dt
        self.dampening = dampening

    def step(self, u, f):
        """
        Conduct one iteration of the integrator.

        :arg `np.array` u: A (`nnodes`, 3) array containing the displacments
            of all nodes.
        :arg `np.array` f: A (`nnodes`, 3) array containing the components of
            the force acting on each node.

        :returns: The new displacements after integration.
        :rtype: `np.array`
        """
        return u + self.dt * f * self.dampening
