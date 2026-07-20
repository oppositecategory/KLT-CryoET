import numpy as np
from numpy.testing import assert_allclose

from kltpicker_3d.utils import (
    generate_uniform_radial_sampling_points,
    trigonometric_interpolation,
)


def test_radial_sampling_uses_angular_frequencies_and_nyquist_ball():
    size = 33
    bandlimit = np.pi

    points, shell_ids, counts = generate_uniform_radial_sampling_points(
        size,
        bandlimit,
    )

    frequencies = np.fft.fftshift(
        2 * np.pi * np.fft.fftfreq(size, d=1.0)
    )
    wx, wy, wz = np.meshgrid(
        frequencies,
        frequencies,
        frequencies,
        indexing="ij",
    )
    radius = np.sqrt(wx**2 + wy**2 + wz**2)
    inside = radius <= bandlimit + np.finfo(float).eps * bandlimit

    assert points.size == (size + 1) // 2
    assert_allclose(points[[0, -1]], [0.0, bandlimit])
    assert np.all(shell_ids[inside] >= 0)
    assert np.all(shell_ids[~inside] == -1)
    assert counts.sum() == np.count_nonzero(inside)
    assert np.all(counts > 0)


def test_radial_sampling_excludes_cube_corners():
    _, shell_ids, _ = generate_uniform_radial_sampling_points(33, np.pi)

    assert shell_ids[0, 0, 0] == -1
    assert shell_ids[-1, -1, -1] == -1


def test_trigonometric_interpolation_reproduces_even_and_odd_nodes():
    for sample_count in (16, 17):
        nodes = np.linspace(0, np.pi, sample_count)
        values = np.exp(-nodes**2)

        interpolated = trigonometric_interpolation(nodes, values, nodes)

        dtype_tolerance = 5 * np.finfo(np.asarray(interpolated).dtype).eps
        assert_allclose(
            interpolated,
            values,
            rtol=dtype_tolerance,
            atol=dtype_tolerance,
        )
