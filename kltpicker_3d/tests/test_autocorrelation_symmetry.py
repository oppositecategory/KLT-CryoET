import numpy as np
from numpy.testing import assert_allclose

from kltpicker_3d.spectral_estimation import (
    cfftn,
    create_autocorrelation_tensor,
)


def test_radial_autocorrelation_contains_both_support_endpoints():
    max_d = 2
    size = 5
    center = size - 1
    squared_distances = np.array([0, 1, 2, 3, 4])
    radial_values = np.array([10.0, 4.0, 3.0, 2.0, 1.0])

    tensor = np.asarray(
        create_autocorrelation_tensor(
            radial_values,
            squared_distances,
            size,
            max_d,
        )
    )

    assert tensor[center - max_d, center, center] == 1.0
    assert tensor[center + max_d, center, center] == 1.0
    assert_allclose(
        tensor,
        np.flip(tensor, axis=(0, 1, 2)),
        rtol=0,
        atol=0,
    )


def test_symmetric_autocorrelation_has_real_fourier_transform():
    tensor = create_autocorrelation_tensor(
        r=np.array([8.0, 3.0, 1.0]),
        dists=np.array([0, 1, 4]),
        N=5,
        max_d=2,
    )

    spectrum = np.asarray(cfftn(tensor))
    relative_imaginary_residual = (
        np.max(np.abs(spectrum.imag))
        / np.max(np.abs(spectrum.real))
    )
    assert relative_imaginary_residual < 10 * np.finfo(spectrum.real.dtype).eps
