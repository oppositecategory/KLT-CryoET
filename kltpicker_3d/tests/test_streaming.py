import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from kltpicker_3d.spectral_estimation import (
    estimate_isotropic_powerspectrum_tensor,
)
from kltpicker_3d.streaming import (
    ArrayVolumeSource,
    StreamingRpsdConfig,
    ShardedRpsdEstimator,
    default_psd_patch_size,
    estimate_patch_rpsd,
    suggest_core_patch_shape,
)
from kltpicker_3d.utils import (
    generate_uniform_radial_sampling_points,
    radial_average_jax,
)


def test_streamed_rpsds_match_in_memory_patch_reduction():
    rng = np.random.default_rng(1701)
    volume = rng.standard_normal((16, 12, 18)).astype(np.float32)
    patch_size = 5
    config = StreamingRpsdConfig(
        patch_size=patch_size,
        core_patch_shape=(2, 1, 2),
        patches_per_microbatch=3,
        halo=1,
    )

    result = ShardedRpsdEstimator(
        ArrayVolumeSource(volume),
        config,
        devices=(jax.devices()[0],),
    ).extract()

    blocks = np.asarray(volume.shape) // patch_size
    cropped = volume[
        : blocks[0] * patch_size,
        : blocks[1] * patch_size,
        : blocks[2] * patch_size,
    ]
    patches = cropped.reshape(
        blocks[0],
        patch_size,
        blocks[1],
        patch_size,
        blocks[2],
        patch_size,
    ).transpose(0, 2, 4, 1, 3, 5)
    patches = jnp.asarray(
        patches.reshape(-1, patch_size, patch_size, patch_size)
    )
    patches = patches - jnp.mean(
        patches,
        axis=(1, 2, 3),
        keepdims=True,
    )
    max_distance = int(np.floor(0.3 * patch_size))
    spectra = jax.vmap(
        estimate_isotropic_powerspectrum_tensor,
        in_axes=(0, None),
    )(patches, max_distance)
    radial_points, shell_ids, counts = generate_uniform_radial_sampling_points(
        2 * patch_size - 1,
        np.pi,
    )
    expected_rpsds = jax.vmap(
        radial_average_jax,
        in_axes=(0, None, None, None),
    )(
        spectra,
        shell_ids,
        counts,
        radial_points.size,
    )

    assert result.patch_grid_shape == (3, 2, 3)
    assert result.patch_size == patch_size
    assert_array_equal(result.radial_points, radial_points)
    assert_allclose(
        result.variances,
        np.var(np.asarray(patches), axis=(1, 2, 3)),
        rtol=1e-6,
    )
    assert_allclose(result.rpsds, np.asarray(expected_rpsds), rtol=2e-5, atol=2e-5)
    direct_rpsd, direct_variance = estimate_patch_rpsd(
        patches[0],
        shell_ids,
        counts,
        max_distance,
    )
    assert_allclose(direct_rpsd, expected_rpsds[0])
    assert_allclose(direct_variance, result.variances[0], rtol=1e-6)


def test_default_patch_and_memory_shape_helpers():
    assert default_psd_patch_size(7.0, 1.0) == 5
    assert suggest_core_patch_shape(
        patch_size=5,
        device_memory_bytes=2 * 4 * 50**3,
        memory_fraction=0.5,
        resident_volume_copies=1,
    ) == (10, 10, 10)


def test_particle_geometry_constructor_caps_core_to_volume():
    volume = np.zeros((16, 11, 26), dtype=np.float32)
    extractor = ShardedRpsdEstimator.from_particle_geometry(
        ArrayVolumeSource(volume),
        particle_diameter=7.0,
        mgscale=1.0,
        devices=(jax.devices()[0],),
        device_memory_bytes=2 * 4 * 50**3,
        memory_fraction=0.5,
        resident_volume_copies=1,
    )

    assert extractor.config.patch_size == 5
    assert extractor.config.core_patch_shape == (3, 2, 5)
    assert extractor.host_buffer_bytes > 0


@pytest.mark.skipif(
    jax.device_count() < 2,
    reason="requires multiple visible JAX devices",
)
def test_subvolume_axis_is_sharded_across_multiple_devices():
    rng = np.random.default_rng(23)
    volume = rng.standard_normal((10, 5, 40)).astype(np.float32)
    devices = tuple(jax.devices()[:8])
    result = ShardedRpsdEstimator(
        ArrayVolumeSource(volume),
        StreamingRpsdConfig(
            patch_size=5,
            core_patch_shape=(1, 1, 1),
            patches_per_microbatch=1,
        ),
        devices=devices,
    ).extract()

    patches = volume.reshape(2, 5, 1, 5, 8, 5).transpose(
        0,
        2,
        4,
        1,
        3,
        5,
    )
    expected_variances = np.var(
        patches.reshape(-1, 5, 5, 5),
        axis=(1, 2, 3),
    )
    assert_allclose(result.variances, expected_variances, rtol=1e-6)
    assert np.isfinite(result.rpsds).all()
