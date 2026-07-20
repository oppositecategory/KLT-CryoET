import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from kltpicker_3d.utils import (
    expand_spherical_harmonic_templates,
    radial_mode_truncation_index,
)


def test_spherical_harmonics_use_colatitude_not_elevation():
    x = np.array([0.0, 1.0, 0.0])
    y = np.zeros(3)
    z = np.array([1.0, 0.0, -1.0])
    radial_templates = np.ones((1, 3))

    templates, _, m_values = expand_spherical_harmonic_templates(
        radial_templates,
        orders=np.array([1]),
        x=x,
        y=y,
        z=z,
    )

    y_10 = templates[np.flatnonzero(m_values == 0)[0]]
    expected_scale = np.sqrt(3 / (4 * np.pi))
    assert_allclose(
        y_10,
        [expected_scale, 0.0, -expected_scale],
        atol=1e-14,
    )


def test_complete_multiplets_have_2ell_plus_1_nonzero_modes():
    x = np.array([1.0, 1.0, -1.0, -1.0])
    y = np.array([1.0, -1.0, 1.0, -1.0])
    z = np.array([0.5, -0.5, 1.5, -1.5])
    orders = np.array([0, 1, 2])
    radial_templates = np.ones((orders.size, x.size))

    templates, radial_indices, m_values = (
        expand_spherical_harmonic_templates(
            radial_templates,
            orders,
            x,
            y,
            z,
        )
    )

    assert templates.shape == (1 + 3 + 5, x.size)
    assert np.all(np.linalg.norm(templates, axis=1) > 0)
    assert_array_equal(
        np.bincount(radial_indices),
        2 * orders + 1,
    )

    # Spherical-harmonic addition theorem:
    # sum_m |Y_ell^m|^2 = (2*ell+1)/(4*pi).
    for radial_index, ell in enumerate(orders):
        modes = templates[radial_indices == radial_index]
        assert_allclose(
            np.sum(np.abs(modes) ** 2, axis=0),
            (2 * ell + 1) / (4 * np.pi),
            rtol=1e-13,
            atol=1e-14,
        )
        assert_array_equal(
            m_values[radial_indices == radial_index],
            np.arange(-ell, ell + 1),
        )


def test_energy_cutoff_counts_angular_degeneracy():
    # Without multiplicity, the first eigenvalue contains >90% of the radial
    # sum. With the eleven ell=5 modes included, both multiplets are required.
    assert (
        radial_mode_truncation_index(
            eigvals=np.array([0.8, 0.03]),
            orders=np.array([0, 5]),
            energy_fraction=0.9,
        )
        == 2
    )
