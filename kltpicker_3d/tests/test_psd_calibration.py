import numpy as np
from numpy.testing import assert_allclose

from kltpicker_3d.utils import (
    calibrate_radial_psds,
    construct_finite_whitening_filter,
    prewhiten_tomogram,
    radial_psd_to_variance,
)


def _scale_to_variance(points, spectrum, target_variance):
    return (
        spectrum
        * target_variance
        / radial_psd_to_variance(points, spectrum)
    )


def test_radial_psd_variance_for_constant_spectrum():
    points = np.linspace(0.0, np.pi, 2001)
    spectrum = np.full_like(points, 6.0)

    # The Nyquist-sphere integral plus constant extension through the FFT-cube
    # corners must preserve a flat PSD exactly.
    assert_allclose(
        radial_psd_to_variance(points, spectrum),
        6.0,
        rtol=1e-6,
    )


def test_psd_calibration_resolves_als_scale_and_baseline_offset():
    points = np.linspace(0.0, np.pi, 1001)
    particle_shape = np.exp(-(points / 1.1) ** 2)
    noise_shape = 0.7 + 0.15 * (points / np.pi) ** 2

    true_particle = _scale_to_variance(
        points,
        particle_shape,
        target_variance=2.0,
    )
    true_noise = _scale_to_variance(
        points,
        noise_shape,
        target_variance=3.0,
    )

    # Mimic the ALS ambiguity: an arbitrary clean-spectrum scale and a
    # multiple of that component absorbed into the noise baseline.
    raw_particle = 4.0 * true_particle
    raw_noise = true_noise + 0.2 * raw_particle

    calibrated_particle, calibrated_noise = calibrate_radial_psds(
        points,
        raw_particle,
        raw_noise,
        noise_variance=3.0,
        mean_patch_variance=5.0,
    )

    assert_allclose(calibrated_particle, true_particle, rtol=1e-12)
    assert_allclose(calibrated_noise, true_noise, rtol=1e-12)
    assert_allclose(
        radial_psd_to_variance(points, calibrated_particle),
        2.0,
        rtol=1e-12,
    )
    assert_allclose(
        radial_psd_to_variance(points, calibrated_noise),
        3.0,
        rtol=1e-12,
    )


def test_prewhitening_reverses_a_known_isotropic_noise_color():
    rng = np.random.default_rng(1701)
    shape = (15, 17, 19)
    white = rng.standard_normal(shape)
    white -= white.mean()

    points = np.linspace(0.0, np.pi, 129)
    noise_psd = 0.25 + 3.0 * np.exp(-(points / 0.9) ** 2)

    frequencies = [
        2 * np.pi * np.fft.fftfreq(size)
        for size in shape
    ]
    wx, wy, wz = np.meshgrid(*frequencies, indexing="ij")
    radius = np.minimum(
        np.sqrt(wx**2 + wy**2 + wz**2),
        points[-1],
    )
    noise_tensor = np.interp(radius, points, noise_psd)
    colored = np.fft.ifftn(
        np.fft.fftn(white) * np.sqrt(noise_tensor)
    ).real

    whitened = np.asarray(
        prewhiten_tomogram(
            colored,
            points,
            noise_psd,
            regularization_fraction=0.0,
        )
    )
    expected = white / np.linalg.norm(white)

    assert_allclose(whitened, expected, rtol=2e-5, atol=2e-7)


def test_finite_whitening_filter_for_flat_psd_is_centered_impulse():
    points = np.linspace(0.0, np.pi, 17)
    noise_psd = np.full_like(points, 4.0)

    whitening_filter = construct_finite_whitening_filter(
        points,
        noise_psd,
        patch_size=7,
        support_radius=2,
        regularization_fraction=0.0,
    )

    expected = np.zeros((5, 5, 5))
    expected[2, 2, 2] = 0.5
    assert_allclose(whitening_filter, expected, atol=1e-15)


def test_finite_whitening_filter_has_requested_spherical_support():
    points = np.linspace(0.0, np.pi, 33)
    noise_psd = 0.5 + np.exp(-(points / 0.8) ** 2)
    support_radius = 3

    whitening_filter = construct_finite_whitening_filter(
        points,
        noise_psd,
        patch_size=9,
        support_radius=support_radius,
    )

    axis = np.arange(-support_radius, support_radius + 1)
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
    outside_support = z**2 + y**2 + x**2 > support_radius**2

    assert whitening_filter.shape == (7, 7, 7)
    assert np.isfinite(whitening_filter).all()
    assert_allclose(
        whitening_filter,
        np.flip(whitening_filter, axis=(0, 1, 2)),
        atol=1e-15,
    )
    assert_allclose(whitening_filter[outside_support], 0)
