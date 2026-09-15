import numpy as np
from numpy.testing import assert_allclose
from scipy.special import roots_legendre

from kltpicker_3d.tomogram import _interpolate_particle_psd_for_fredholm


def test_fredholm_psd_interpolation_preserves_nonnegative_radial_mass():
    points = np.linspace(0.0, np.pi, 93)
    spectrum = np.exp(-((points - 0.35) / 0.18) ** 2)
    spectrum[points > 0.9] = 0
    legendre_nodes, legendre_weights = roots_legendre(150)
    nodes = np.pi / 2 * (legendre_nodes + 1)

    interpolated = _interpolate_particle_psd_for_fredholm(
        points,
        spectrum,
        nodes,
        legendre_weights,
    )

    source_integrand = spectrum * points**2
    source_mass = np.sum(
        0.5
        * (source_integrand[:-1] + source_integrand[1:])
        * np.diff(points)
    )
    interpolated_mass = np.pi / 2 * np.sum(
        legendre_weights * interpolated * nodes**2
    )
    assert np.all(interpolated >= 0)
    assert_allclose(interpolated_mass, source_mass, rtol=2e-15, atol=1e-15)
