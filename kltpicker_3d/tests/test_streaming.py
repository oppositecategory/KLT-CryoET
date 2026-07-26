import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.signal import fftconvolve

from kltpicker_3d.spectral_estimation import (
    estimate_isotropic_powerspectrum_tensor,
)
from kltpicker_3d.streaming import (
    ArrayVolumeSource,
    MultiGPUSubvolumeProcessor,
    default_psd_patch_size,
    estimate_patch_rpsd,
    extract_streamed_rpsds,
    suggest_core_patch_shape,
)
from kltpicker_3d.utils import (
    generate_uniform_radial_sampling_points,
    radial_average_jax,
)


def _rpsd_processor(volume, patch_size, core_patch_shape, devices=None):
    patch_grid = tuple(size // patch_size for size in volume.shape)
    return MultiGPUSubvolumeProcessor(
        ArrayVolumeSource(volume),
        domain_shape=tuple(size * patch_size for size in patch_grid),
        core_shape=tuple(size * patch_size for size in core_patch_shape),
        devices=(jax.devices()[0],) if devices is None else devices,
    )


def test_multi_gpu_processor_maps_fixed_subvolume_function():
    volume = np.arange(5 * 4 * 3, dtype=np.float32).reshape(5, 4, 3)
    processor = MultiGPUSubvolumeProcessor(
        ArrayVolumeSource(volume),
        domain_shape=volume.shape,
        core_shape=(3, 3, 2),
        devices=(jax.devices()[0],),
    )

    def transform_subvolume(
        loaded_subvolume,
        scale,
        *,
        core_shape,
        halo,
        offset,
    ):
        core = jax.lax.dynamic_slice(
            loaded_subvolume,
            (halo, halo, halo),
            core_shape,
        )
        transformed = core * scale + offset
        return transformed, jnp.sum(transformed)

    assembled = np.empty_like(volume)
    observed_sums = []
    outputs = processor.map(
        transform_subvolume,
        np.float32(2),
        halo=1,
        static_kwargs={
            "core_shape": processor.core_shape,
            "halo": 1,
            "offset": 1.0,
        },
        description="processor test",
    )
    for regions, (blocks, block_sums) in outputs:
        for slot, region in enumerate(regions):
            if region is None:
                continue
            destination = tuple(
                slice(start, stop)
                for start, stop in zip(
                    region.start,
                    region.stop,
                    strict=True,
                )
            )
            valid_shape = tuple(
                stop - start
                for start, stop in zip(
                    region.start,
                    region.stop,
                    strict=True,
                )
            )
            local = tuple(slice(0, size) for size in valid_shape)
            assembled[destination] = blocks[slot][local]
            observed_sums.append(block_sums[slot])

    assert processor.loaded_shape(1) == (5, 5, 4)
    assert processor.subvolume_grid_shape == (2, 2, 2)
    assert processor.round_count == 8
    assert processor.host_buffer_bytes(1) > 0
    assert_allclose(assembled, volume * 2 + 1)
    assert np.isfinite(observed_sums).all()


def test_streamed_rpsds_match_in_memory_patch_reduction():
    rng = np.random.default_rng(1701)
    volume = rng.standard_normal((16, 12, 18)).astype(np.float32)
    patch_size = 5
    processor = _rpsd_processor(
        volume,
        patch_size,
        core_patch_shape=(2, 1, 2),
    )
    assert processor.subvolume_grid_shape == (2, 2, 2)
    assert processor.round_count == 8
    result = extract_streamed_rpsds(
        processor,
        patch_size,
        patches_per_microbatch=3,
    )

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


def test_whitened_streaming_matches_filtering_complete_volume_first():
    rng = np.random.default_rng(90210)
    volume = rng.standard_normal((16, 12, 18)).astype(np.float32)
    axis = np.arange(-1, 2)
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
    whitening_filter = np.exp(-(z**2 + y**2 + x**2)).astype(np.float32)
    whitening_filter /= whitening_filter.sum()
    processor = _rpsd_processor(
        volume,
        patch_size=5,
        core_patch_shape=(2, 1, 2),
    )
    streamed = extract_streamed_rpsds(
        processor,
        patch_size=5,
        patches_per_microbatch=1,
        spatial_filter=whitening_filter,
    )

    complete_filtered_volume = fftconvolve(
        volume,
        whitening_filter,
        mode="same",
    ).astype(np.float32)
    reference = extract_streamed_rpsds(
        _rpsd_processor(
            complete_filtered_volume,
            patch_size=5,
            core_patch_shape=(2, 1, 2),
        ),
        patch_size=5,
        patches_per_microbatch=1,
    )

    assert_allclose(streamed.variances, reference.variances, rtol=2e-5, atol=2e-6)
    assert_allclose(streamed.rpsds, reference.rpsds, rtol=2e-5, atol=2e-5)


def test_default_patch_and_memory_shape_helpers():
    assert default_psd_patch_size(7.0, 1.0) == 5
    assert suggest_core_patch_shape(
        patch_size=5,
        device_memory_bytes=2 * 4 * 50**3,
        memory_fraction=0.5,
        resident_volume_copies=1,
    ) == (10, 10, 10)


def test_memory_planning_caps_core_to_volume():
    volume = np.zeros((16, 11, 26), dtype=np.float32)
    patch_size = default_psd_patch_size(7.0, 1.0)
    patch_grid = tuple(size // patch_size for size in volume.shape)
    suggested = suggest_core_patch_shape(
        patch_size,
        device_memory_bytes=2 * 4 * 50**3,
        memory_fraction=0.5,
        resident_volume_copies=1,
    )
    core_patch_shape = tuple(
        min(planned, available)
        for planned, available in zip(suggested, patch_grid, strict=True)
    )

    assert patch_size == 5
    assert core_patch_shape == (3, 2, 5)


def test_processor_is_reused_for_functions_with_different_halos():
    rng = np.random.default_rng(7)
    volume = rng.standard_normal((10, 10, 10)).astype(np.float32)
    spatial_filter = np.ones((3, 3, 3), dtype=np.float32)
    processor = _rpsd_processor(volume, 5, (1, 1, 1))

    initial = extract_streamed_rpsds(
        processor,
        5,
        patches_per_microbatch=1,
    )
    filtered = extract_streamed_rpsds(
        processor,
        5,
        patches_per_microbatch=1,
        spatial_filter=spatial_filter,
    )

    assert processor.loaded_shape(0) == (5, 5, 5)
    assert processor.loaded_shape(1) == (7, 7, 7)
    assert initial.rpsds.shape == filtered.rpsds.shape


@pytest.mark.skipif(
    jax.device_count() < 2,
    reason="requires multiple visible JAX devices",
)
def test_subvolume_axis_is_sharded_across_multiple_devices():
    rng = np.random.default_rng(23)
    volume = rng.standard_normal((10, 5, 40)).astype(np.float32)
    devices = tuple(jax.devices()[:8])
    result = extract_streamed_rpsds(
        _rpsd_processor(
            volume,
            patch_size=5,
            core_patch_shape=(1, 1, 1),
            devices=devices,
        ),
        patch_size=5,
        patches_per_microbatch=1,
    )

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
