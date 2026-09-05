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
    distributed_block_qr_score_parameters,
    extract_score_candidates,
    next_cufft_fast_length,
    orthogonal_klt_score_parameters,
    plan_cufft_fft_shape,
    plan_template_fft_batch,
    ranked_candidate_nms_3d,
)
from kltpicker_3d.streaming import ArrayVolumeSource
from kltpicker_3d.utils import construct_finite_bandpass_filter


def test_finite_bandpass_rejects_constant_and_has_bounded_support():
    kernel = construct_finite_bandpass_filter(
        patch_size=9,
        support_radius=3,
        low_fraction=0.05,
        high_fraction=0.05,
    )

    assert kernel.shape == (7, 7, 7)
    assert_allclose(np.sum(kernel), 0, atol=1e-12)
    grid = np.indices(kernel.shape) - 3
    outside = np.sum(grid**2, axis=0) > 3**2
    assert_array_equal(kernel[outside], 0)


def test_candidate_top_k_reports_full_count_and_retains_only_capacity():
    scores = np.zeros((7, 7, 7), dtype=np.float32)
    scores[2, 2, 2] = 3
    scores[4, 4, 4] = 2

    coordinates, values, count = extract_score_candidates(
        jnp.asarray(scores),
        jnp.full(3, 5, dtype=jnp.int32),
        core_shape=(5, 5, 5),
        source_shape=(20, 20, 20),
        template_radius=0,
        candidate_capacity=1,
    )

    assert int(count) == 2
    assert_array_equal(np.asarray(coordinates), np.array([[1, 1, 1]]))
    assert_allclose(np.asarray(values), np.array([3], dtype=np.float32))


def test_distributed_block_qr_preserves_each_signal_covariance():
    rng = np.random.default_rng(123)
    templates = (
        rng.standard_normal((4, 3, 3, 3))
        + 1j * rng.standard_normal((4, 3, 3, 3))
    ).astype(np.complex64)
    eigenvalues = np.array([4.0, 1.5, 3.0, 0.5], dtype=np.float32)
    orders = np.array([0, 0, 1, 1])
    m_values = np.array([0, 0, 0, 0])

    basis, weights, offset, transformed_eigenvalues = (
        distributed_block_qr_score_parameters(
            templates,
            eigenvalues,
            orders,
            m_values,
            noise_variance=0.8,
            devices=(jax.devices()[0],),
        )
    )

    for indices in (np.array([0, 1]), np.array([2, 3])):
        original = templates[indices].reshape(2, -1).T
        transformed = basis[indices].reshape(2, -1).T
        expected_covariance = (
            original * eigenvalues[indices][None, :]
        ) @ original.conj().T
        actual_covariance = (
            transformed * transformed_eigenvalues[indices][None, :]
        ) @ transformed.conj().T
        assert_allclose(actual_covariance, expected_covariance, rtol=2e-5, atol=2e-5)
        assert_allclose(
            transformed.conj().T @ transformed,
            np.eye(2),
            rtol=2e-5,
            atol=2e-5,
        )
    expected_weights = 1 / 0.8 - 1 / (0.8 + transformed_eigenvalues)
    assert_allclose(weights, expected_weights, rtol=2e-5)
    assert_allclose(
        offset,
        np.sum(np.log1p(transformed_eigenvalues / 0.8)),
        rtol=2e-5,
    )


def test_nonnegative_m_scoring_matches_explicit_conjugate_pairs():
    rng = np.random.default_rng(2026)
    positive = (
        rng.standard_normal((2, 3, 3, 3))
        + 1j * rng.standard_normal((2, 3, 3, 3))
    ).astype(np.complex64)
    zero = rng.standard_normal((2, 3, 3, 3)).astype(np.complex64)
    templates = np.empty((6, 3, 3, 3), dtype=np.complex64)
    for radial_index in range(2):
        start = 3 * radial_index
        templates[start] = -np.conj(positive[radial_index])
        templates[start + 1] = zero[radial_index]
        templates[start + 2] = positive[radial_index]

    eigenvalues = np.repeat(
        np.array([3.0, 0.75], dtype=np.float32),
        3,
    )
    orders = np.ones(6, dtype=np.int64)
    m_values = np.tile(np.array([-1, 0, 1], dtype=np.int64), 2)
    kernels, effective_weights, offset, transformed_eigenvalues = (
        distributed_block_qr_score_parameters(
            templates,
            eigenvalues,
            orders,
            m_values,
            noise_variance=0.8,
            devices=(jax.devices()[0],),
            host_qr=True,
        )
    )

    assert kernels.shape == (4, 3, 3, 3)
    representative_m = m_values[m_values >= 0]
    assert_array_equal(representative_m, np.array([0, 1, 0, 1]))
    multiplicities = np.where(representative_m == 0, 1, 2)
    expected_base_weights = 1 / 0.8 - 1 / (
        0.8 + transformed_eigenvalues
    )
    assert_allclose(
        effective_weights,
        multiplicities * expected_base_weights,
        rtol=2e-5,
    )
    assert_allclose(
        offset,
        np.sum(
            multiplicities
            * np.log1p(transformed_eigenvalues / 0.8)
        ),
        rtol=2e-5,
    )

    zero_indices = np.flatnonzero(representative_m == 0)
    positive_indices = np.flatnonzero(representative_m > 0)
    expanded_kernels = np.concatenate(
        (
            kernels[zero_indices],
            kernels[positive_indices],
            np.conj(kernels[positive_indices]),
        )
    )
    expanded_weights = np.concatenate(
        (
            effective_weights[zero_indices],
            effective_weights[positive_indices] / 2,
            effective_weights[positive_indices] / 2,
        )
    )
    loaded = rng.standard_normal((5, 5, 5)).astype(np.float32)
    whitening_filter = np.ones((1, 1, 1), dtype=np.float32)
    compressed_score = compute_klt_score_block(
        jnp.asarray(loaded),
        jnp.asarray(whitening_filter),
        jnp.asarray(kernels),
        jnp.asarray(effective_weights),
        jnp.asarray(offset),
        core_shape=(3, 3, 3),
        whitening_radius=0,
        score_halo=0,
    )
    expanded_score = compute_klt_score_block(
        jnp.asarray(loaded),
        jnp.asarray(whitening_filter),
        jnp.asarray(expanded_kernels),
        jnp.asarray(expanded_weights),
        jnp.asarray(offset),
        core_shape=(3, 3, 3),
        whitening_radius=0,
        score_halo=0,
    )

    assert_allclose(compressed_score, expanded_score, rtol=2e-5, atol=2e-5)


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


def test_cufft_fast_length_and_shape_planning():
    assert next_cufft_fast_length(358) == 360
    assert next_cufft_fast_length(360) == 360
    assert plan_cufft_fft_shape((358, 359, 360)) == (360, 360, 360)
    assert plan_cufft_fft_shape((358, 359, 360), (360, 364, 375)) == (
        360,
        364,
        375,
    )


def test_fast_fft_padding_preserves_fused_valid_scores():
    rng = np.random.default_rng(45)
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

    arguments = (
        jnp.asarray(loaded),
        jnp.asarray(whitening_filter),
        jnp.asarray(templates),
        jnp.asarray(normalization),
        jnp.asarray(weights),
        jnp.asarray(offset),
    )
    options = dict(
        core_shape=(5, 5, 5),
        whitening_radius=1,
        template_radius=1,
        score_halo=1,
        template_batch_size=2,
        axis_name=None,
    )
    exact = compute_fused_klt_score_shard(
        *arguments,
        **options,
        fft_shape=(11, 11, 11),
    )
    padded = compute_fused_klt_score_shard(
        *arguments,
        **options,
        fft_shape=(12, 12, 12),
    )

    assert_allclose(padded, exact, rtol=2e-5, atol=2e-4)


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


def test_spatial_hash_nms_matches_quadratic_reference():
    rng = np.random.default_rng(18)
    candidates = np.column_stack(
        (
            rng.integers(0, 40, size=(300, 3)),
            rng.standard_normal(300),
        )
    ).astype(np.float64)

    def quadratic(radius: float, max_picks: int) -> np.ndarray:
        order = np.lexsort(
            (
                candidates[:, 2],
                candidates[:, 1],
                candidates[:, 0],
                -candidates[:, 3],
            )
        )
        accepted = []
        for candidate in candidates[order]:
            if accepted:
                differences = np.asarray(accepted)[:, :3] - candidate[:3]
                if np.any(np.sum(differences**2, axis=1) <= radius**2):
                    continue
            accepted.append(candidate)
            if len(accepted) == max_picks:
                break
        return np.asarray(accepted)

    for radius in (0, 1, 3.5, 8):
        for max_picks in (1, 20, len(candidates)):
            assert_array_equal(
                ranked_candidate_nms_3d(
                    candidates,
                    radius=radius,
                    max_picks=max_picks,
                ),
                quadratic(radius, max_picks),
            )


def test_streamed_candidates_match_complete_volume_scoring():
    rng = np.random.default_rng(91)
    volume = rng.standard_normal((13, 14, 15)).astype(np.float32)
    whitening_filter = np.ones((3, 3, 3), dtype=np.float32) / 27
    templates = rng.standard_normal((7, 3, 3, 3)).astype(np.float32)
    template_eigenvalues = np.linspace(2.0, 0.5, 7, dtype=np.float32)
    noise_variance = 0.8

    detector = MultiGPUKLTParticleDetector3D(
        ArrayVolumeSource(volume),
        particle_diameter=5,
        mgscale=1,
        num_particles=20,
        whitening_support_radius=1,
        bandpass_low_fraction=0,
        bandpass_high_fraction=0,
        devices=tuple(jax.devices()),
        core_patch_shape=(2, 2, 2),
        candidate_capacity_per_subvolume=6**3,
        psd_patch_size=3,
        fredholm_radius=1,
        template_side=3,
        score_template_chunk_size=1,
    )
    detector.model.eigvals = template_eigenvalues
    detector.model.template_orders = np.zeros(7, dtype=np.int64)
    detector.model.template_m_values = np.zeros(7, dtype=np.int64)
    kernels, weights, offset, _ = detector.prepare_score_filters(
        templates,
        noise_variance,
    )
    streamed = detector.score_candidates(
        templates,
        noise_variance,
        whitening_filter,
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
        bandpass_low_fraction=0,
        bandpass_high_fraction=0,
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
    assert np.all(detector.model.template_m_values >= 0)
    assert detector.templates.shape[0] == detector.score_templates.shape[0]
    assert_allclose(
        np.sum(detector.model.template_multiplicities),
        detector.model.retained_template_count,
    )
