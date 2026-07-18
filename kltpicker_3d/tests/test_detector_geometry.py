import jax.numpy as jnp
import numpy as np
import pytest

from kltpicker_3d.tomogram import (
    KLTParticleDetector3D,
    _extract_nonoverlapping_patches,
)


def make_detector(**geometry):
    return KLTParticleDetector3D(
        jnp.zeros((40, 40, 40), dtype=jnp.float32),
        particle_diameter=33,
        mgscale=1,
        num_particles=1,
        **geometry,
    )


def test_optimized_geometry_defaults_are_independent():
    detector = make_detector()

    assert detector.patch_size == 25
    assert np.isclose(detector.fredholm_radius_voxels, 13.2)
    assert detector.template_side == 29
    assert detector.nms_radius_voxels == 16.5


def test_geometry_can_be_configured_independently():
    detector = make_detector(
        psd_patch_size=21,
        fredholm_radius=5.5,
        template_side=17,
    )

    assert detector.patch_size == 21
    assert detector.fredholm_radius_voxels == 5.5
    assert detector.template_side == 17


def test_nms_radius_can_be_configured_independently():
    detector = make_detector(nms_radius=16.5)

    assert detector.nms_radius_voxels == 16.5


def test_nms_radius_must_be_nonnegative():
    with pytest.raises(ValueError, match="nms_radius"):
        make_detector(nms_radius=-1)


def test_rectangular_patch_extraction_uses_each_axis_independently():
    volume = np.arange(6 * 10 * 14).reshape(6, 10, 14)

    patches = _extract_nonoverlapping_patches(volume, patch_size=2)

    assert patches.shape == (3 * 5 * 7, 2, 2, 2)
    np.testing.assert_array_equal(patches[0], volume[:2, :2, :2])
    np.testing.assert_array_equal(patches[-1], volume[4:6, 8:10, 12:14])


def test_patch_extraction_requires_every_axis_to_fit():
    with pytest.raises(ValueError, match="all tomogram dimensions"):
        _extract_nonoverlapping_patches(
            np.zeros((2, 5, 5)),
            patch_size=3,
        )


def test_template_grid_must_contain_fredholm_ball():
    with pytest.raises(ValueError, match="too small"):
        make_detector(
            fredholm_radius=6.6,
            template_side=13,
        )
