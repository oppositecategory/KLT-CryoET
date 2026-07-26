import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy import ndimage
from scipy.signal import fftconvolve

from kltpicker_3d.multi_gpu import (
    MultiGPUKLTParticleDetector3D,
    compute_fused_klt_score_shard,
    compute_klt_score_block,
    construct_klt_score_filters,
    orthogonal_klt_score_parameters,
    plan_template_fft_batch,
    ranked_candidate_nms_3d,
)
from kltpicker_3d.streaming import ArrayVolumeSource


def test_fused_batched_fft_matches_sequential_convolution():
    rng = np.random.default_rng(44)
    core_shape = (5, 5, 5)
    loaded = rng.standard_normal((11, 11, 11)).astype(np.float32)
    whitening_filter = rng.standard_normal((3, 3, 3)).astype(np.float32)
    templates = (
        rng.standard_normal((2, 3, 3, 3))
        + 1j * rng.standard_normal((2, 3, 3, 3))
    ).astype(np.complex64)
    normalization, weights, offset, _ = orthogonal_klt_score_parameters(
        templates,
        np.array([2.0, 0.75]),
        0.8,
    )
    normalized_templates = templates * normalization[:, None, None, None]

    sequential = compute_klt_score_block(
        jnp.asarray(loaded),
        jnp.asarray(whitening_filter),
        jnp.asarray(normalized_templates),
        jnp.asarray(weights),
        jnp.asarray(offset),
        core_shape=core_shape,
        whitening_radius=1,
        score_halo=1,
    )
    fused = compute_fused_klt_score_shard(
        jnp.asarray(loaded),
        jnp.asarray(whitening_filter),
        jnp.asarray(templates),
        jnp.asarray(normalization),
        jnp.asarray(weights),
        jnp.asarray(offset),
        core_shape=core_shape,
        whitening_radius=1,
        template_radius=1,
        score_halo=1,
        template_batch_size=2,
        axis_name=None,
    )

    assert_allclose(fused, sequential, rtol=1e-5, atol=2e-4)


def test_fft_batch_planner_accounts_for_resident_template_shard():
    plan = plan_template_fft_batch(
        (32, 32, 32),
        (24, 24, 24),
        (9, 9, 9),
        templates_per_device=10,
        device_memory_bytes=64 * 2**20,
        memory_fraction=0.8,
    )

    assert plan["batch_size"] >= 1
    assert plan["estimated_peak_bytes"] <= plan["budget_bytes"]


def test_global_candidate_nms_is_independent_of_device_completion_order():
    candidates = np.array(
        [
            [4, 5, 5, 10],
            [5, 5, 5, 9],
            [15, 5, 5, 8],
            [15, 6, 5, 7],
            [25, 5, 5, 8],
        ],
        dtype=np.float64,
    )
    expected = np.array(
        [
            [4, 5, 5, 10],
            [15, 5, 5, 8],
            [25, 5, 5, 8],
        ],
        dtype=np.float64,
    )

    forward = ranked_candidate_nms_3d(candidates, radius=2, max_picks=10)
    shuffled = ranked_candidate_nms_3d(
        candidates[[3, 1, 4, 0, 2]],
        radius=2,
        max_picks=10,
    )

    assert_array_equal(forward, expected)
    assert_array_equal(shuffled, expected)


def test_streamed_candidates_match_complete_volume_scoring():
    rng = np.random.default_rng(91)
    volume = rng.standard_normal((13, 14, 15)).astype(np.float32)
    whitening_filter = np.ones((3, 3, 3), dtype=np.float32) / 27
    templates = rng.standard_normal((2, 3, 3, 3)).astype(np.float32)
    template_eigenvalues = np.array([2.0, 0.75], dtype=np.float32)
    noise_variance = 0.8

    detector = MultiGPUKLTParticleDetector3D(
        ArrayVolumeSource(volume),
        particle_diameter=5,
        mgscale=1,
        num_particles=20,
        whitening_support_radius=1,
        devices=tuple(jax.devices()),
        core_patch_shape=(2, 2, 2),
        candidate_capacity_per_subvolume=6**3,
        psd_patch_size=3,
        fredholm_radius=1,
        template_side=3,
    )
    detector.model.eigvals = template_eigenvalues
    streamed = detector.score_candidates(
        templates,
        noise_variance,
        whitening_filter,
    )

    kernels, weights, offset = construct_klt_score_filters(
        templates,
        template_eigenvalues,
        noise_variance,
    )
    whitened = fftconvolve(volume, whitening_filter, mode="same")
    complete_score = np.zeros(
        tuple(size - 2 for size in volume.shape),
        dtype=np.float32,
    )
    for kernel, weight in zip(kernels, weights, strict=True):
        response = fftconvolve(
            whitened,
            np.conj(np.flip(kernel, axis=(0, 1, 2))),
            mode="valid",
        )
        complete_score += weight * np.square(np.abs(response))
    complete_score -= offset

    maximum = ndimage.maximum_filter(
        complete_score,
        size=3,
        mode="constant",
        cval=-np.inf,
    )
    minimum = ndimage.minimum_filter(
        complete_score,
        size=3,
        mode="constant",
        cval=np.inf,
    )
    mask = (
        np.isfinite(complete_score)
        & (complete_score == maximum)
        & (complete_score > minimum)
    )
    reference_coordinates = np.argwhere(mask) + 1
    reference_scores = complete_score[mask]
    reference = np.column_stack((reference_coordinates, reference_scores))

    streamed = streamed[np.lexsort(streamed[:, :3].T[::-1])]
    reference = reference[np.lexsort(reference[:, :3].T[::-1])]
    assert_array_equal(streamed[:, :3], reference[:, :3])
    assert_allclose(streamed[:, 3], reference[:, 3], rtol=2e-4, atol=2e-4)


def test_complete_multi_gpu_detector_executes_on_streamed_volume():
    rng = np.random.default_rng(1701)
    shape = (20, 20, 20)
    grid = np.indices(shape, dtype=np.float32)
    volume = 0.3 * rng.standard_normal(shape).astype(np.float32)
    for center in ((6, 6, 6), (14, 14, 13)):
        radius_squared = sum(
            (grid[axis] - center[axis]) ** 2 for axis in range(3)
        )
        volume += np.exp(-radius_squared / (2 * 1.25**2)).astype(np.float32)

    detector = MultiGPUKLTParticleDetector3D(
        ArrayVolumeSource(volume),
        particle_diameter=7,
        mgscale=1,
        num_particles=2,
        whitening_support_radius=1,
        devices=(jax.devices()[0],),
        core_patch_shape=(2, 2, 2),
        candidate_capacity_per_subvolume=10**3,
        patches_per_microbatch=1,
        legendre_order=8,
        threshold=-1,
        max_iter=30,
        max_order=3,
    )
    count, particles = detector.process_tomogram()

    assert count == 2
    assert particles.shape == (2, 4)
    assert np.isfinite(particles).all()
    assert detector.initial_rpsds.rpsds.shape[0] == 4**3
    assert detector.whitened_rpsds.rpsds.shape == detector.initial_rpsds.rpsds.shape
    assert detector.templates.ndim == 4
