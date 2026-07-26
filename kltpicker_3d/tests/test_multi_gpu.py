import jax
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy import ndimage
from scipy.signal import fftconvolve

from kltpicker_3d.multi_gpu import (
    MultiGPUKLTParticleDetector3D,
    construct_klt_score_filters,
    ranked_candidate_nms_3d,
)
from kltpicker_3d.streaming import ArrayVolumeSource


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
