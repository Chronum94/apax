import numpy as np
import pytest

from apax.md.simulate import check_minimum_image
from apax.utils.space import perpendicular_widths


def test_perpendicular_widths_orthorhombic():
    np.testing.assert_allclose(perpendicular_widths(np.diag([4.0, 5.0, 6.0])), [4, 5, 6])


def test_perpendicular_widths_tilted_cell():
    # ASE cell rows; the tilted c vector leaves only 3.5 A between the ab planes
    cell = np.array([[10.0, 0, 0], [0, 10.0, 0], [9.0, 0, 3.5]])
    widths = perpendicular_widths(cell.T)
    assert np.isclose(widths.min(), 3.5)
    assert np.linalg.norm(cell, axis=1).min() > 9


@pytest.mark.parametrize("r_max, ok", [(1.7, True), (1.76, False), (3.0, False)])
def test_check_minimum_image(r_max, ok):
    cell = np.array([[10.0, 0, 0], [0, 10.0, 0], [9.0, 0, 3.5]])
    if ok:
        check_minimum_image(cell.T, r_max)
    else:
        with pytest.raises(ValueError, match="minimum image"):
            check_minimum_image(cell.T, r_max)
