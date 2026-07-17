import numpy as np
from numpy.testing import assert_allclose
from scipy.special import roots_legendre, spherical_jn

from kltpicker_3d.fredholm_solver import (
    _build_radial_fredholm_matrices,
    solve_radial_fredholm_equation,
)


def test_kernel_uses_spatial_times_frequency_grid():
    order = 2
    quadrature_order = 16
    spatial_radius = 3.5
    bandlimit = np.pi

    nodes, weights = roots_legendre(quadrature_order)
    frequencies = (bandlimit / 2) * (nodes + 1)
    radii = (spatial_radius / 2) * (nodes + 1)
    spectrum = np.exp(-frequencies**2)

    kernel, _ = _build_radial_fredholm_matrices(
        spectrum,
        order,
        spatial_radius,
        bandlimit,
        quadrature_order,
    )

    basis = (
        4
        * np.pi
        * (1j**order)
        * spherical_jn(order, np.outer(radii, frequencies))
    )
    frequency_measure = (
        (bandlimit / 2)
        * weights
        * spectrum
        * frequencies**2
    )
    expected = (basis * frequency_measure[None, :]) @ basis.conj().T

    assert_allclose(kernel, expected, rtol=1e-12, atol=1e-12)


def test_radial_eigenfunctions_are_weighted_orthonormal():
    quadrature_order = 24
    spatial_radius = 4.0
    bandlimit = np.pi
    nodes, _ = roots_legendre(quadrature_order)
    frequencies = (bandlimit / 2) * (nodes + 1)
    spectrum = np.exp(-0.5 * frequencies**2)

    eigvals, eigfuncs, weights = solve_radial_fredholm_equation(
        spectrum,
        N=1,
        a=spatial_radius,
        c=bandlimit,
        K=quadrature_order,
    )

    active = np.flatnonzero(eigvals > 0)[:6]
    gram = eigfuncs[:, active].conj().T @ weights @ eigfuncs[:, active]
    assert_allclose(gram, np.eye(active.size), rtol=1e-10, atol=1e-10)
