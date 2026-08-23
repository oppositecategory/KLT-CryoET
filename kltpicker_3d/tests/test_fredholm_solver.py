import numpy as np
from numpy.testing import assert_allclose
from scipy.special import roots_legendre, spherical_jn

from kltpicker_3d.fredholm_solver import (
    INVERSE_FOURIER_NORMALIZATION_3D,
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
        INVERSE_FOURIER_NORMALIZATION_3D
        * (bandlimit / 2)
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


def test_complete_angular_spectrum_has_expected_covariance_trace():
    quadrature_order = 48
    spatial_radius = 1.5
    bandlimit = np.pi
    spectrum_level = 2.0
    nodes, weights = roots_legendre(quadrature_order)
    frequencies = (bandlimit / 2) * (nodes + 1)
    spectrum = np.full(quadrature_order, spectrum_level)

    covariance_at_zero = (
        4
        * np.pi
        * INVERSE_FOURIER_NORMALIZATION_3D
        * (bandlimit / 2)
        * np.sum(weights * spectrum * frequencies**2)
    )
    expected_trace = 4 * np.pi * spatial_radius**3 / 3 * covariance_at_zero

    observed_trace = 0.0
    for order in range(18):
        eigenvalues, _, _ = solve_radial_fredholm_equation(
            spectrum,
            N=order,
            a=spatial_radius,
            c=bandlimit,
            K=quadrature_order,
        )
        observed_trace += (2 * order + 1) * np.sum(eigenvalues[eigenvalues > 0])

    assert_allclose(observed_trace, expected_trace, rtol=2e-10, atol=2e-10)
