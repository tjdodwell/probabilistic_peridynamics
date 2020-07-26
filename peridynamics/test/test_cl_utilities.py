"""Tests for the utilities module."""
import numpy as np
from peridynamics.utilities import (read_array, _calc_midpoint_gradient,
                                    calc_displacement_scale, calc_build_time)
import pytest
import pathlib