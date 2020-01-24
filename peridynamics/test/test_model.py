"""
Tests for the model class
"""
from ..SeqPeriVectorized import SeqModel as Model
from ..SeqPeriVectorized import DimensionalityError
import pytest


class TestDimension:
    def test_2d(self):
        model = Model(dimensions=2)

        assert model.mesh_elements.connectivity == 'triangle'
        assert model.mesh_elements.boundary == 'line'

    def test_3d(self):
        model = Model(dimensions=3)

        assert model.mesh_elements.connectivity == 'tetrahedron'
        assert model.mesh_elements.boundary == 'triangle'

    @pytest.mark.parametrize("dimensions", [1, 4])
    def test_dimensionality_error(self, dimensions):
        with pytest.raises(DimensionalityError):
            Model(dimensions=dimensions)
