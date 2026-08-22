"""Single-host multi-GPU orchestration for three-dimensional KLT detection."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from kltpicker_3d.alt_least_squares import alternating_least_squares_solver
from kltpicker_3d.streaming import (
    MultiGPUSubvolumeProcessor,
    PaddingMode,
    RpsdExtractionResult,
    VolumeSource,
    apply_finite_spatial_filter,
    default_psd_patch_size,
    estimate_device_memory_limit,
    extract_streamed_rpsds,
    spatial_filter_radius,
    suggest_core_patch_shape,
)
from kltpicker_3d.tomogram import KLTParticleDetector3D
from kltpicker_3d.utils import (
    calibrate_radial_psds,
    construct_finite_bandpass_filter,
    construct_finite_whitening_filter,
)

_ALS_CONVERGENCE_TOLERANCE = 1e-4
_NOISE_PATCH_FRACTION = 0.25
_LOCAL_MAXIMUM_RADIUS = 1
_TEMPLATE_AXIS_NAME = "template_device"
_DEFAULT_SCORE_MEMORY_FRACTION = 0.8
_SCORE_FIXED_COMPLEX_ARRAYS = 5
# Batched 3-D cuFFT plans require substantial backend workspace in addition
# to the visible padded filters, spectra, and responses. Eight complex-grid
# equivalents is deliberately conservative; the original factor of three
# selected B=3 for EMPIAR-10045 and cuFFT then requested another 2.42 GiB.
_SCORE_BATCH_COMPLEX_ARRAYS = 8
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CalibratedRpsdModel:
    """ALS particle/noise spectra calibrated to streamed patch variances."""

    particle_psd: npt.NDArray[np.float64]
    noise_psd: npt.NDArray[np.float64]
    noise_variance: float


def fit_streamed_rpsds(
    extraction: RpsdExtractionResult,
    *,
    max_iterations: int,
    convergence_tolerance: float = _ALS_CONVERGENCE_TOLERANCE,
    device: jax.Device | None = None,
) -> CalibratedRpsdModel:
    """Fit and variance-calibrate the ALS model from streamed RPSDs."""
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if convergence_tolerance <= 0:
        raise ValueError("convergence_tolerance must be positive")
    if extraction.rpsds.ndim != 2 or extraction.variances.ndim != 1:
        raise ValueError("invalid streamed RPSD result")
    if extraction.rpsds.shape[0] != extraction.variances.size:
        raise ValueError("RPSDs and variances must have equal sample counts")

    variances = np.asarray(extraction.variances, dtype=np.float64)
    noise_patch_count = max(
        1,
        int(np.floor(_NOISE_PATCH_FRACTION * variances.size)),
    )
    noise_variance = float(np.mean(np.partition(variances, noise_patch_count - 1)[
        :noise_patch_count
    ]))
    mean_patch_variance = float(np.mean(variances))

    selected_device = jax.devices()[0] if device is None else device
    with jax.default_device(selected_device):
        factorization = alternating_least_squares_solver(
            jnp.asarray(extraction.rpsds),
            max_iterations,
            convergence_tolerance,
        )
        particle_psd = np.asarray(factorization.gamma)
        noise_psd = np.asarray(factorization.v)
    particle_psd, noise_psd = calibrate_radial_psds(
        extraction.radial_points,
        particle_psd,
        noise_psd,
        noise_variance,
        mean_patch_variance,
    )
    return CalibratedRpsdModel(
        particle_psd=particle_psd,
        noise_psd=noise_psd,
        noise_variance=noise_variance,
    )


def orthogonal_klt_score_parameters(
    templates: npt.ArrayLike,
    template_eigenvalues: npt.ArrayLike,
    noise_variance: float,
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    np.float32,
    npt.NDArray[np.float64],
]:
    """Return normalization, weights, and offset for orthogonal KLT modes.

    Fredholm eigenfunctions are orthonormal in the continuous weighted
    problem. Each sampled Cartesian template is normalized independently;
    the corresponding covariance eigenvalue is rescaled so this operation
    leaves its rank-one covariance contribution unchanged. Cross-template
    voxel correlations are intentionally ignored.
    """
    templates = np.asarray(templates)
    eigenvalues = np.asarray(template_eigenvalues, dtype=np.float64)
    if templates.ndim != 4 or templates.shape[0] < 1:
        raise ValueError("templates must have shape (modes, z, y, x)")
    if len(set(templates.shape[1:])) != 1 or templates.shape[1] % 2 == 0:
        raise ValueError("templates must have an odd cubic spatial shape")
    if eigenvalues.shape != (templates.shape[0],):
        raise ValueError("each template must have one eigenvalue")
    if not np.all(np.isfinite(eigenvalues)) or np.any(eigenvalues < 0):
        raise ValueError("template eigenvalues must be finite and nonnegative")
    if noise_variance <= 0:
        raise ValueError("noise_variance must be positive")
    if not np.isfinite(noise_variance):
        raise ValueError("noise_variance must be finite")

    norm_squared = np.empty(templates.shape[0], dtype=np.float64)
    for index, template in enumerate(templates):
        flattened = np.asarray(template).reshape(-1)
        norm_squared[index] = float(np.real(np.vdot(flattened, flattened)))
    if not np.all(np.isfinite(norm_squared)) or np.any(norm_squared <= 0):
        raise ValueError("templates must have positive finite voxel norms")

    normalization = np.asarray(1.0 / np.sqrt(norm_squared), dtype=np.float32)
    adjusted_eigenvalues = eigenvalues * norm_squared
    covariance_values = noise_variance + adjusted_eigenvalues
    weights = 1.0 / noise_variance - 1.0 / covariance_values
    offset = np.sum(np.log1p(adjusted_eigenvalues / noise_variance))
    return (
        normalization,
        np.asarray(weights, dtype=np.float32),
        np.float32(offset),
        adjusted_eigenvalues,
    )


def _qr_score_block(
    flattened_templates: jax.Array,
    template_eigenvalues: jax.Array,
    noise_variance: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Transform one fixed ``(ell, m)`` block into likelihood eigenmodes."""
    orthonormal_basis, triangular_factor = jnp.linalg.qr(
        flattened_templates,
        mode="reduced",
    )
    signal_covariance = (
        triangular_factor * template_eigenvalues[None, :]
    ) @ jnp.conj(triangular_factor.T)
    signal_covariance = 0.5 * (
        signal_covariance + jnp.conj(signal_covariance.T)
    )
    signal_eigenvalues, covariance_eigenvectors = jnp.linalg.eigh(
        signal_covariance
    )
    signal_eigenvalues = jnp.maximum(signal_eigenvalues, 0)[::-1]
    covariance_eigenvectors = covariance_eigenvectors[:, ::-1]
    covariance_eigenvalues = noise_variance + signal_eigenvalues
    score_weights = 1.0 / noise_variance - 1.0 / covariance_eigenvalues
    score_offset = jnp.sum(jnp.log(covariance_eigenvalues / noise_variance))
    score_basis = orthonormal_basis @ covariance_eigenvectors
    return (
        score_basis.T,
        score_weights,
        score_offset,
        signal_eigenvalues,
    )


def _host_qr_score_block(
    flattened_templates: npt.NDArray[np.complex64],
    template_eigenvalues: npt.NDArray[np.float32],
    noise_variance: float,
) -> tuple[
    npt.NDArray[np.complex64],
    npt.NDArray[np.float32],
    np.float32,
    npt.NDArray[np.float64],
]:
    """Compute one block with host LAPACK when GPU complex QR is unavailable."""
    orthonormal_basis, triangular_factor = np.linalg.qr(
        flattened_templates,
        mode="reduced",
    )
    signal_covariance = (
        triangular_factor * template_eigenvalues[None, :]
    ) @ triangular_factor.conj().T
    signal_covariance = 0.5 * (
        signal_covariance + signal_covariance.conj().T
    )
    signal_eigenvalues, covariance_eigenvectors = np.linalg.eigh(
        signal_covariance
    )
    signal_eigenvalues = np.maximum(signal_eigenvalues, 0)[::-1]
    covariance_eigenvectors = covariance_eigenvectors[:, ::-1]
    covariance_eigenvalues = noise_variance + signal_eigenvalues
    score_weights = 1.0 / noise_variance - 1.0 / covariance_eigenvalues
    score_offset = np.sum(np.log(covariance_eigenvalues / noise_variance))
    score_basis = orthonormal_basis @ covariance_eigenvectors
    return (
        np.asarray(score_basis.T, dtype=np.complex64),
        np.asarray(score_weights, dtype=np.float32),
        np.float32(score_offset),
        np.asarray(signal_eigenvalues, dtype=np.float64),
    )


def distributed_block_qr_score_parameters(
    templates: npt.ArrayLike,
    template_eigenvalues: npt.ArrayLike,
    template_orders: npt.ArrayLike,
    template_m_values: npt.ArrayLike,
    noise_variance: float,
    *,
    devices: Sequence[jax.Device] | None = None,
    output: npt.NDArray[np.complex64] | None = None,
    host_qr: bool = False,
) -> tuple[
    npt.NDArray[np.complex64],
    npt.NDArray[np.float32],
    np.float32,
    npt.NDArray[np.float64],
]:
    """Compute independent QR likelihood bases for every ``(ell, m)`` block.

    Same-width blocks are dispatched independently across the selected devices.
    ``host_qr`` uses NumPy/LAPACK for systems whose GPU backend does not support
    complex QR. No Gram-matrix precheck is performed: block QR is the scoring
    definition. Cross-block orthogonality follows the spherical-harmonic
    construction.
    """
    template_array = np.asanyarray(templates)
    eigenvalues = np.asarray(template_eigenvalues, dtype=np.float32)
    orders = np.asarray(template_orders, dtype=np.int64)
    m_values = np.asarray(template_m_values, dtype=np.int64)
    if template_array.ndim != 4 or template_array.shape[0] < 1:
        raise ValueError("templates must have shape (modes, z, y, x)")
    mode_count = template_array.shape[0]
    if any(values.shape != (mode_count,) for values in (eigenvalues, orders, m_values)):
        raise ValueError("eigenvalues, orders, and m values must match templates")
    if noise_variance <= 0 or not np.isfinite(noise_variance):
        raise ValueError("noise_variance must be positive and finite")
    selected_devices = tuple(jax.devices() if devices is None else devices)
    if not selected_devices:
        raise ValueError("at least one JAX device is required")

    if output is None:
        score_templates = np.empty(template_array.shape, dtype=np.complex64)
    else:
        if output.shape != template_array.shape or output.dtype != np.complex64:
            raise ValueError("output must be complex64 with the template shape")
        score_templates = output
    score_weights = np.empty(mode_count, dtype=np.float32)
    signal_eigenvalues = np.empty(mode_count, dtype=np.float64)
    voxel_count = int(np.prod(template_array.shape[1:]))

    blocks_by_width: dict[int, list[npt.NDArray[np.int64]]] = {}
    for order, m_value in sorted(set(zip(orders.tolist(), m_values.tolist()))):
        indices = np.flatnonzero((orders == order) & (m_values == m_value))
        blocks_by_width.setdefault(indices.size, []).append(indices)

    total_offset = 0.0
    for width, blocks in sorted(blocks_by_width.items()):
        LOGGER.info(
            "Block QR: radial width=%d | angular blocks=%d | devices=%d",
            width,
            len(blocks),
            len(selected_devices),
        )
        if host_qr:
            for block_index, indices in enumerate(blocks, start=1):
                LOGGER.info(
                    "Host block QR %d/%d for radial width=%d",
                    block_index,
                    len(blocks),
                    width,
                )
                host_templates = np.asarray(
                    template_array[indices],
                    dtype=np.complex64,
                ).reshape(width, voxel_count).T
                basis_block, weight_block, offset, eigenvalue_block = (
                    _host_qr_score_block(
                        host_templates,
                        eigenvalues[indices],
                        float(noise_variance),
                    )
                )
                score_templates[indices] = basis_block.reshape(
                    width,
                    *template_array.shape[1:],
                )
                score_weights[indices] = weight_block
                signal_eigenvalues[indices] = eigenvalue_block
                total_offset += float(offset)
            continue
        compiled_qr = jax.jit(
            partial(_qr_score_block, noise_variance=float(noise_variance)),
        )
        for round_start in range(0, len(blocks), len(selected_devices)):
            round_blocks = blocks[round_start : round_start + len(selected_devices)]
            LOGGER.info(
                "Block QR round %d/%d: processing %d block(s)",
                round_start // len(selected_devices) + 1,
                (len(blocks) + len(selected_devices) - 1)
                // len(selected_devices),
                len(round_blocks),
            )
            pending_results = []
            for slot, indices in enumerate(round_blocks):
                host_templates = np.asarray(
                    template_array[indices],
                    dtype=np.complex64,
                ).reshape(width, voxel_count).T
                device = selected_devices[slot]
                device_templates = jax.device_put(host_templates, device)
                device_eigenvalues = jax.device_put(eigenvalues[indices], device)
                pending_results.append(
                    (
                        indices,
                        compiled_qr(device_templates, device_eigenvalues),
                    )
                )

            # JAX dispatch is asynchronous. Launch every independent block
            # before gathering any result so the selected GPUs work in
            # parallel without a replicated pmap computation or collectives.
            for indices, device_result in pending_results:
                basis_block, weight_block, offset, eigenvalue_block = (
                    np.asarray(value) for value in device_result
                )
                score_templates[indices] = basis_block.reshape(
                    width,
                    *template_array.shape[1:],
                )
                score_weights[indices] = weight_block
                signal_eigenvalues[indices] = eigenvalue_block
                total_offset += float(offset)
    return (
        score_templates,
        score_weights,
        np.float32(total_offset),
        signal_eigenvalues,
    )


def construct_klt_score_filters(
    templates: npt.ArrayLike,
    template_eigenvalues: npt.ArrayLike,
    noise_variance: float,
) -> tuple[
    npt.NDArray[np.complex64],
    npt.NDArray[np.float32],
    np.float32,
]:
    """Materialize normalized score filters for small in-memory problems."""
    template_array = np.asarray(templates)
    normalization, weights, offset, _ = orthogonal_klt_score_parameters(
        template_array,
        template_eigenvalues,
        noise_variance,
    )
    kernels = np.asarray(
        template_array * normalization[:, None, None, None],
        dtype=np.complex64,
    )
    return kernels, weights, offset


def plan_template_fft_batch(
    loaded_shape: tuple[int, int, int],
    core_shape: tuple[int, int, int],
    template_shape: tuple[int, int, int],
    templates_per_device: int,
    device_memory_bytes: int,
    *,
    memory_fraction: float = _DEFAULT_SCORE_MEMORY_FRACTION,
) -> dict[str, int | float]:
    """Estimate a conservative resident batch for fused complex FFT scoring."""
    if any(size < 1 for size in loaded_shape + core_shape + template_shape):
        raise ValueError("scoring shapes must be positive")
    if templates_per_device < 1:
        raise ValueError("templates_per_device must be positive")
    if device_memory_bytes < 1:
        raise ValueError("device_memory_bytes must be positive")
    if not 0 < memory_fraction <= 1:
        raise ValueError("score memory_fraction must lie in (0, 1]")

    real_bytes = np.dtype(np.float32).itemsize
    complex_bytes = np.dtype(np.complex64).itemsize
    loaded_voxels = int(np.prod(loaded_shape))
    output_voxels = int(
        np.prod(tuple(size + 2 * _LOCAL_MAXIMUM_RADIUS for size in core_shape))
    )
    template_voxels = int(np.prod(template_shape))
    budget_bytes = int(device_memory_bytes * memory_fraction)
    resident_template_bytes = (
        templates_per_device * template_voxels * complex_bytes
    )
    fixed_bytes = (
        loaded_voxels
        * (real_bytes + _SCORE_FIXED_COMPLEX_ARRAYS * complex_bytes)
        + 2 * output_voxels * real_bytes
        + resident_template_bytes
    )
    bytes_per_batch_template = (
        loaded_voxels * _SCORE_BATCH_COMPLEX_ARRAYS * complex_bytes
    )
    available_bytes = budget_bytes - fixed_bytes
    batch_size = min(
        templates_per_device,
        max(0, available_bytes // bytes_per_batch_template),
    )
    if batch_size < 1:
        raise ValueError(
            "estimated score memory cannot hold one FFT template batch; "
            "reduce core_patch_shape or increase score_memory_fraction"
        )
    return {
        "batch_size": int(batch_size),
        "budget_bytes": budget_bytes,
        "fixed_bytes": fixed_bytes,
        "bytes_per_batch_template": bytes_per_batch_template,
        "estimated_peak_bytes": int(
            fixed_bytes + batch_size * bytes_per_batch_template
        ),
        "memory_fraction": float(memory_fraction),
    }


def compute_klt_score_block(
    loaded_subvolume: jax.Array,
    whitening_filter: jax.Array,
    score_kernels: jax.Array,
    score_weights: jax.Array,
    score_offset: jax.Array,
    *,
    core_shape: tuple[int, int, int],
    whitening_radius: int,
    score_halo: int,
) -> jax.Array:
    """Return KLT scores for one core plus a small local-maximum halo."""
    whitened = apply_finite_spatial_filter(
        loaded_subvolume,
        whitening_filter,
    )
    output_shape = tuple(size + 2 * score_halo for size in core_shape)
    initial_score = jnp.zeros(
        output_shape,
        dtype=jnp.result_type(loaded_subvolume, jnp.float32),
    )

    def accumulate(index: jax.Array, score: jax.Array) -> jax.Array:
        kernel = jnp.conj(jnp.flip(score_kernels[index], axis=(0, 1, 2)))
        response = jax.scipy.signal.fftconvolve(
            whitened,
            kernel,
            mode="valid",
        )
        owned_response = jax.lax.dynamic_slice(
            response,
            (whitening_radius,) * 3,
            output_shape,
        )
        return score + score_weights[index] * jnp.square(jnp.abs(owned_response))

    score = jax.lax.fori_loop(
        0,
        score_kernels.shape[0],
        accumulate,
        initial_score,
    )
    return score - score_offset


def _centered_filter_spectrum(
    spatial_filters: jax.Array,
    fft_shape: tuple[int, int, int],
) -> jax.Array:
    """Embed centered filters on an overlap-save grid and transform them."""
    filters = jnp.asarray(spatial_filters)
    if filters.ndim == 3:
        filters = filters[None, ...]
    filter_shape = filters.shape[-3:]
    padding = (
        (0, 0),
        (0, fft_shape[0] - filter_shape[0]),
        (0, fft_shape[1] - filter_shape[1]),
        (0, fft_shape[2] - filter_shape[2]),
    )
    padded = jnp.pad(filters, padding)
    radii = tuple((size - 1) // 2 for size in filter_shape)
    padded = jnp.roll(
        padded,
        shift=tuple(-radius for radius in radii),
        axis=(-3, -2, -1),
    )
    return jnp.fft.fftn(padded, axes=(-3, -2, -1))


def compute_fused_klt_score_shard(
    loaded_subvolume: jax.Array,
    whitening_filter: jax.Array,
    local_templates: jax.Array,
    local_normalization: jax.Array,
    local_weights: jax.Array,
    score_offset: jax.Array,
    *,
    core_shape: tuple[int, int, int],
    whitening_radius: int,
    template_radius: int,
    score_halo: int,
    template_batch_size: int,
    axis_name: str | None,
) -> jax.Array:
    """Accumulate one template shard using a shared fused subvolume FFT."""
    if local_templates.shape[0] % template_batch_size:
        raise ValueError("local template shard must contain complete batches")
    fft_shape = tuple(int(size) for size in loaded_subvolume.shape)
    output_shape = tuple(size + 2 * score_halo for size in core_shape)
    crop_start = whitening_radius + template_radius

    volume_spectrum = jnp.fft.fftn(loaded_subvolume)
    whitening_spectrum = _centered_filter_spectrum(
        whitening_filter,
        fft_shape,
    )[0]
    whitened_spectrum = volume_spectrum * whitening_spectrum
    initial_score = jnp.zeros(output_shape, dtype=jnp.float32)
    batch_count = local_templates.shape[0] // template_batch_size

    def accumulate(batch_index: jax.Array, score: jax.Array) -> jax.Array:
        start = batch_index * template_batch_size
        templates = jax.lax.dynamic_slice_in_dim(
            local_templates,
            start,
            template_batch_size,
            axis=0,
        )
        normalization = jax.lax.dynamic_slice_in_dim(
            local_normalization,
            start,
            template_batch_size,
            axis=0,
        )
        weights = jax.lax.dynamic_slice_in_dim(
            local_weights,
            start,
            template_batch_size,
            axis=0,
        )
        correlation_filters = jnp.conj(
            jnp.flip(
                templates * normalization[:, None, None, None],
                axis=(-3, -2, -1),
            )
        )
        template_spectra = _centered_filter_spectrum(
            correlation_filters,
            fft_shape,
        )
        responses = jnp.fft.ifftn(
            whitened_spectrum[None, ...] * template_spectra,
            axes=(-3, -2, -1),
        )
        owned = jax.lax.dynamic_slice(
            responses,
            (0, crop_start, crop_start, crop_start),
            (template_batch_size, *output_shape),
        )
        return score + jnp.sum(
            weights[:, None, None, None] * jnp.square(jnp.abs(owned)),
            axis=0,
        )

    partial_score = jax.lax.fori_loop(
        0,
        batch_count,
        accumulate,
        initial_score,
    )
    score = (
        partial_score
        if axis_name is None
        else jax.lax.psum(partial_score, axis_name)
    )
    return score - score_offset


def extract_score_candidates(
    haloed_scores: jax.Array,
    region_start: jax.Array,
    *,
    core_shape: tuple[int, int, int],
    source_shape: tuple[int, int, int],
    template_radius: int,
    candidate_capacity: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return the ranked 3x3x3 local maxima from one owned score core."""
    if candidate_capacity > int(np.prod(core_shape)):
        raise ValueError("candidate_capacity cannot exceed the core voxel count")
    score_halo = _LOCAL_MAXIMUM_RADIUS
    valid_center_mask = jnp.ones(haloed_scores.shape, dtype=bool)
    for axis in range(3):
        global_axis = (
            jnp.arange(haloed_scores.shape[axis])
            + region_start[axis]
            - score_halo
        )
        axis_valid = (
            (global_axis >= template_radius)
            & (global_axis < source_shape[axis] - template_radius)
        )
        reshape = [1, 1, 1]
        reshape[axis] = haloed_scores.shape[axis]
        valid_center_mask &= axis_valid.reshape(reshape)
    haloed_scores = jnp.where(valid_center_mask, haloed_scores, -jnp.inf)
    neighborhood_max = jax.lax.reduce_window(
        haloed_scores,
        -jnp.inf,
        jax.lax.max,
        (3, 3, 3),
        (1, 1, 1),
        "SAME",
    )
    neighborhood_min = jax.lax.reduce_window(
        haloed_scores,
        jnp.inf,
        jax.lax.min,
        (3, 3, 3),
        (1, 1, 1),
        "SAME",
    )
    local_scores = jax.lax.dynamic_slice(
        haloed_scores,
        (score_halo,) * 3,
        core_shape,
    )
    local_maximum = jax.lax.dynamic_slice(
        neighborhood_max,
        (score_halo,) * 3,
        core_shape,
    )
    local_minimum = jax.lax.dynamic_slice(
        neighborhood_min,
        (score_halo,) * 3,
        core_shape,
    )
    candidate_mask = (
        jnp.isfinite(local_scores)
        & (local_scores == local_maximum)
        & (local_scores > local_minimum)
    )
    candidate_count = jnp.sum(candidate_mask, dtype=jnp.int32)
    ranked_scores, flat_indices = jax.lax.top_k(
        jnp.where(candidate_mask, local_scores, -jnp.inf).ravel(),
        candidate_capacity,
    )
    coordinates = jnp.stack(
        jnp.unravel_index(flat_indices, core_shape),
        axis=1,
    )
    return coordinates, ranked_scores, candidate_count


def score_subvolume_candidates(
    loaded_subvolume: jax.Array,
    region_start: jax.Array,
    whitening_filter: jax.Array,
    score_kernels: jax.Array,
    score_weights: jax.Array,
    score_offset: jax.Array,
    *,
    core_shape: tuple[int, int, int],
    source_shape: tuple[int, int, int],
    whitening_radius: int,
    template_radius: int,
    candidate_capacity: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Score one core and return its highest 3x3x3 local maxima."""
    score_halo = _LOCAL_MAXIMUM_RADIUS
    haloed_scores = compute_klt_score_block(
        loaded_subvolume,
        whitening_filter,
        score_kernels,
        score_weights,
        score_offset,
        core_shape=core_shape,
        whitening_radius=whitening_radius,
        score_halo=score_halo,
    )
    return extract_score_candidates(
        haloed_scores,
        region_start,
        core_shape=core_shape,
        source_shape=source_shape,
        template_radius=template_radius,
        candidate_capacity=candidate_capacity,
    )


def score_template_shards_and_extract_candidates(
    loaded_subvolume: jax.Array,
    region_start: jax.Array,
    whitening_filter: jax.Array,
    local_templates: jax.Array,
    local_normalization: jax.Array,
    local_weights: jax.Array,
    score_offset: jax.Array,
    *,
    core_shape: tuple[int, int, int],
    source_shape: tuple[int, int, int],
    whitening_radius: int,
    template_radius: int,
    candidate_capacity: int,
    template_batch_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """All-reduce template-sharded scores and extract candidates once."""
    haloed_scores = compute_fused_klt_score_shard(
        loaded_subvolume,
        whitening_filter,
        local_templates,
        local_normalization,
        local_weights,
        score_offset,
        core_shape=core_shape,
        whitening_radius=whitening_radius,
        template_radius=template_radius,
        score_halo=_LOCAL_MAXIMUM_RADIUS,
        template_batch_size=template_batch_size,
        axis_name=_TEMPLATE_AXIS_NAME,
    )

    def extract(_: None) -> tuple[jax.Array, jax.Array, jax.Array]:
        return extract_score_candidates(
            haloed_scores,
            region_start,
            core_shape=core_shape,
            source_shape=source_shape,
            template_radius=template_radius,
            candidate_capacity=candidate_capacity,
        )

    def empty(_: None) -> tuple[jax.Array, jax.Array, jax.Array]:
        return (
            jnp.zeros((candidate_capacity, 3), dtype=jnp.int32),
            jnp.full((candidate_capacity,), -jnp.inf, dtype=jnp.float32),
            jnp.asarray(0, dtype=jnp.int32),
        )

    return jax.lax.cond(
        jax.lax.axis_index(_TEMPLATE_AXIS_NAME) == 0,
        extract,
        empty,
        operand=None,
    )


def ranked_candidate_nms_3d(
    candidates: npt.ArrayLike,
    *,
    radius: float,
    max_picks: int,
) -> npt.NDArray[np.float64]:
    """Globally rank candidate rows and greedily apply spherical NMS."""
    candidates = np.asarray(candidates, dtype=np.float64)
    if candidates.ndim != 2 or candidates.shape[1] != 4:
        raise ValueError("candidates must have (z, y, x, score) columns")
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    if max_picks < 0:
        raise ValueError("max_picks must be nonnegative")
    if candidates.size == 0 or max_picks == 0:
        return np.empty((0, 4), dtype=np.float64)

    finite = np.all(np.isfinite(candidates), axis=1)
    candidates = candidates[finite]
    order = np.lexsort(
        (
            candidates[:, 2],
            candidates[:, 1],
            candidates[:, 0],
            -candidates[:, 3],
        )
    )
    ranked = candidates[order]
    accepted = np.empty((min(max_picks, ranked.shape[0]), 4), dtype=np.float64)
    accepted_count = 0
    radius_squared = radius**2
    for candidate in ranked:
        if accepted_count:
            distances_squared = np.sum(
                (accepted[:accepted_count, :3] - candidate[:3]) ** 2,
                axis=1,
            )
            if np.any(distances_squared <= radius_squared):
                continue
        accepted[accepted_count] = candidate
        accepted_count += 1
        if accepted_count == accepted.shape[0]:
            break
    return accepted[:accepted_count]


class MultiGPUKLTParticleDetector3D:
    """Coordinate the complete out-of-core KLT detector on local GPUs."""

    def __init__(
        self,
        source: VolumeSource,
        particle_diameter: float,
        mgscale: float,
        num_particles: int,
        *,
        whitening_support_radius: int,
        bandpass_low_fraction: float = 0.05,
        bandpass_high_fraction: float = 0.05,
        devices: Sequence[jax.Device] | None = None,
        core_patch_shape: tuple[int, int, int] | None = None,
        device_memory_bytes: int | None = None,
        memory_fraction: float = 0.3,
        resident_volume_copies: int = 8,
        patches_per_microbatch: int = 1,
        candidate_capacity_per_subvolume: int = 4096,
        legendre_order: int = 150,
        threshold: float = 0,
        max_iter: int = 500,
        max_order: int = 4,
        template_energy_fraction: float = 0.99,
        max_templates: int | None = 1000,
        psd_patch_size: int | None = None,
        fredholm_radius: float | None = None,
        template_side: int | None = None,
        nms_radius: float | None = None,
        score_template_batch_size: int | None = None,
        score_memory_fraction: float = _DEFAULT_SCORE_MEMORY_FRACTION,
        boundary_mode: PaddingMode = "constant",
    ) -> None:
        """Initialize shared streaming geometry and the KLT template model."""
        selected_devices = tuple(jax.devices() if devices is None else devices)
        if not selected_devices:
            raise ValueError("at least one JAX device is required")
        if whitening_support_radius < 0:
            raise ValueError("whitening_support_radius must be nonnegative")
        if not 0 <= bandpass_low_fraction < 1:
            raise ValueError("bandpass_low_fraction must lie in [0, 1)")
        if not 0 <= bandpass_high_fraction < 1:
            raise ValueError("bandpass_high_fraction must lie in [0, 1)")
        if bandpass_low_fraction + bandpass_high_fraction >= 1:
            raise ValueError("bandpass fractions must sum to less than one")
        if patches_per_microbatch < 1:
            raise ValueError("patches_per_microbatch must be positive")
        if candidate_capacity_per_subvolume < 1:
            raise ValueError("candidate capacity must be positive")
        if score_template_batch_size is not None and score_template_batch_size < 1:
            raise ValueError("score_template_batch_size must be positive")
        if not 0 < score_memory_fraction <= 1:
            raise ValueError("score_memory_fraction must lie in (0, 1]")

        patch_size = (
            default_psd_patch_size(particle_diameter, mgscale)
            if psd_patch_size is None
            else psd_patch_size
        )
        patch_grid_shape = tuple(size // patch_size for size in source.shape)
        if any(size < 1 for size in patch_grid_shape):
            raise ValueError("every source axis must contain at least one patch")

        self.model = KLTParticleDetector3D(
            None,
            particle_diameter,
            mgscale,
            num_particles,
            legendre_order=legendre_order,
            threshold=threshold,
            max_iter=max_iter,
            max_order=max_order,
            template_energy_fraction=template_energy_fraction,
            max_templates=max_templates,
            psd_patch_size=patch_size,
            fredholm_radius=fredholm_radius,
            template_side=template_side,
            nms_radius=nms_radius,
        )
        maximum_halo = (
            whitening_support_radius
            + self.model.template_side // 2
            + _LOCAL_MAXIMUM_RADIUS
        )
        memory_limit = (
            device_memory_bytes
            if device_memory_bytes is not None
            else estimate_device_memory_limit(selected_devices)
        )
        if core_patch_shape is None:
            if memory_limit is None:
                raise ValueError(
                    "device memory is unavailable; provide core_patch_shape "
                    "or device_memory_bytes"
                )
            suggested = suggest_core_patch_shape(
                patch_size,
                memory_limit,
                memory_fraction=memory_fraction,
                resident_volume_copies=resident_volume_copies,
                halo=maximum_halo,
            )
            core_patch_shape = tuple(
                min(planned, available)
                for planned, available in zip(
                    suggested,
                    patch_grid_shape,
                    strict=True,
                )
            )
        if len(core_patch_shape) != 3 or any(size < 1 for size in core_patch_shape):
            raise ValueError("core_patch_shape must contain three positive values")

        self.source = source
        self.devices = selected_devices
        self.patch_size = patch_size
        self.patches_per_microbatch = patches_per_microbatch
        self.whitening_support_radius = whitening_support_radius
        self.bandpass_low_fraction = bandpass_low_fraction
        self.bandpass_high_fraction = bandpass_high_fraction
        self.candidate_capacity_per_subvolume = candidate_capacity_per_subvolume
        self.device_memory_bytes = memory_limit
        self.score_template_batch_size = score_template_batch_size
        self.score_memory_fraction = score_memory_fraction
        self.score_plan: dict[str, int | float] | None = None
        self.processor = MultiGPUSubvolumeProcessor(
            source,
            domain_shape=source.shape,
            core_shape=tuple(size * patch_size for size in core_patch_shape),
            devices=selected_devices,
            boundary_mode=boundary_mode,
        )

        self.initial_rpsds: RpsdExtractionResult | None = None
        self.bandpass_filter: npt.NDArray[np.float32] | None = None
        self.initial_model: CalibratedRpsdModel | None = None
        self.whitening_filter: npt.NDArray[np.float32] | None = None
        self.whitened_rpsds: RpsdExtractionResult | None = None
        self.whitened_model: CalibratedRpsdModel | None = None
        self.templates: npt.NDArray[np.generic] | None = None
        self.score_templates: npt.NDArray[np.complex64] | None = None
        self.template_normalization: npt.NDArray[np.float32] | None = None
        self.score_weights: npt.NDArray[np.float32] | None = None
        self.score_offset: np.float32 | None = None
        self.adjusted_template_eigenvalues: npt.NDArray[np.float64] | None = None
        self.candidates: npt.NDArray[np.float64] | None = None
        self.particles: npt.NDArray[np.float64] | None = None

    def estimate_rpsds(
        self,
        whitening_filter: npt.ArrayLike | None = None,
        *,
        description: str | None = None,
    ) -> RpsdExtractionResult:
        """Run streamed RPSD extraction, optionally with finite whitening."""
        return extract_streamed_rpsds(
            self.processor,
            self.patch_size,
            patches_per_microbatch=self.patches_per_microbatch,
            spatial_filter=whitening_filter,
            description=description,
        )

    def fit_rpsds(self, extraction: RpsdExtractionResult) -> CalibratedRpsdModel:
        """Run ALS on the first selected GPU and calibrate the result."""
        return fit_streamed_rpsds(
            extraction,
            max_iterations=self.model.max_iter,
            device=self.devices[0],
        )

    def build_whitening_filter(
        self,
        noise_psd: npt.ArrayLike,
    ) -> npt.NDArray[np.float32]:
        """Construct the configured finite-support whitening filter."""
        return np.asarray(
            construct_finite_whitening_filter(
                self.model.uniform_points,
                noise_psd,
                self.patch_size,
                self.whitening_support_radius,
                bandpass_low_fraction=self.bandpass_low_fraction,
                bandpass_high_fraction=self.bandpass_high_fraction,
            ),
            dtype=np.float32,
        )

    def build_bandpass_filter(self) -> npt.NDArray[np.float32]:
        """Construct the finite streamed approximation to legacy preprocessing."""
        return np.asarray(
            construct_finite_bandpass_filter(
                self.patch_size,
                self.whitening_support_radius,
                low_fraction=self.bandpass_low_fraction,
                high_fraction=self.bandpass_high_fraction,
            ),
            dtype=np.float32,
        )

    def build_templates(
        self,
        particle_psd: npt.ArrayLike,
    ) -> npt.NDArray[np.generic]:
        """Solve the radial KLT problem and construct complete templates."""
        eigenvalues, eigenfunctions, orders, particle_nodes = (
            self.model._solve_radial_modes(particle_psd)
        )
        templates, _ = self.model.create_gpsf_templates(
            eigenvalues,
            eigenfunctions,
            orders,
            particle_nodes,
        )
        return templates

    def prepare_score_filters(
        self,
        templates: npt.ArrayLike,
        noise_variance: float,
        *,
        output: npt.NDArray[np.complex64] | None = None,
        host_qr: bool = False,
    ) -> tuple[
        npt.NDArray[np.complex64],
        npt.NDArray[np.float32],
        np.float32,
        npt.NDArray[np.float64],
    ]:
        """Prepare distributed ``(ell, m)`` block-QR likelihood filters."""
        if self.model.eigvals is None:
            raise RuntimeError("template eigenvalues have not been initialized")
        if self.model.template_orders is None or self.model.template_m_values is None:
            raise RuntimeError("template angular metadata has not been initialized")
        parameters = distributed_block_qr_score_parameters(
            templates,
            self.model.eigvals,
            self.model.template_orders,
            self.model.template_m_values,
            noise_variance,
            devices=self.devices,
            output=output,
            host_qr=host_qr,
        )
        (
            self.score_templates,
            self.score_weights,
            self.score_offset,
            self.adjusted_template_eigenvalues,
        ) = parameters
        self.template_normalization = np.ones(
            self.score_weights.shape,
            dtype=np.float32,
        )
        return parameters

    def score_candidates(
        self,
        templates: npt.ArrayLike,
        noise_variance: float,
        whitening_filter: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        """Template-shard fused FFT scores and return globally located maxima."""
        if self.model.eigvals is None:
            raise RuntimeError("template eigenvalues have not been initialized")
        raw_templates = np.asanyarray(templates)
        if raw_templates.ndim != 4 or raw_templates.shape[0] < 1:
            raise ValueError("templates must have shape (modes, z, y, x)")
        whitening_filter = np.asarray(whitening_filter, dtype=np.float32)
        whitening_radius = spatial_filter_radius(whitening_filter)
        if (
            self.template_normalization is None
            or self.score_weights is None
            or self.score_offset is None
            or self.adjusted_template_eigenvalues is None
            or self.score_templates is None
            or self.template_normalization.shape != (raw_templates.shape[0],)
        ):
            self.prepare_score_filters(raw_templates, noise_variance)
        templates = np.asanyarray(self.score_templates)
        normalization = self.template_normalization
        score_weights = self.score_weights
        score_offset = self.score_offset

        template_radius = templates.shape[1] // 2
        candidate_capacity = min(
            self.candidate_capacity_per_subvolume,
            int(np.prod(self.processor.core_shape)),
        )
        total_halo = (
            whitening_radius + template_radius + _LOCAL_MAXIMUM_RADIUS
        )
        device_count = len(self.devices)
        templates_per_device = (
            templates.shape[0] + device_count - 1
        ) // device_count
        batch_size = self.score_template_batch_size
        if batch_size is None:
            if self.device_memory_bytes is None:
                batch_size = 1
                self.score_plan = {
                    "batch_size": 1,
                    "memory_fraction": self.score_memory_fraction,
                }
            else:
                self.score_plan = plan_template_fft_batch(
                    self.processor.loaded_shape(total_halo),
                    self.processor.core_shape,
                    tuple(int(size) for size in templates.shape[1:]),
                    templates_per_device,
                    self.device_memory_bytes,
                    memory_fraction=self.score_memory_fraction,
                )
                batch_size = int(self.score_plan["batch_size"])
        batch_size = min(batch_size, templates_per_device)
        padded_templates_per_device = (
            (templates_per_device + batch_size - 1) // batch_size * batch_size
        )

        if self.device_memory_bytes is not None:
            capacity_plan = plan_template_fft_batch(
                self.processor.loaded_shape(total_halo),
                self.processor.core_shape,
                tuple(int(size) for size in templates.shape[1:]),
                padded_templates_per_device,
                self.device_memory_bytes,
                memory_fraction=self.score_memory_fraction,
            )
            if int(capacity_plan["batch_size"]) < batch_size:
                if self.score_template_batch_size is not None:
                    raise ValueError(
                        "configured score_template_batch_size exceeds the "
                        "estimated device memory capacity"
                    )
                batch_size = int(capacity_plan["batch_size"])
                padded_templates_per_device = (
                    (templates_per_device + batch_size - 1)
                    // batch_size
                    * batch_size
                )
                capacity_plan = plan_template_fft_batch(
                    self.processor.loaded_shape(total_halo),
                    self.processor.core_shape,
                    tuple(int(size) for size in templates.shape[1:]),
                    padded_templates_per_device,
                    self.device_memory_bytes,
                    memory_fraction=self.score_memory_fraction,
                )
            capacity_plan["batch_size"] = batch_size
            capacity_plan["templates_per_device"] = padded_templates_per_device
            capacity_plan["template_count"] = int(templates.shape[0])
            capacity_plan["batches_per_device"] = (
                padded_templates_per_device // batch_size
            )
            capacity_plan["subvolume_count"] = self.processor.subvolume_count
            self.score_plan = capacity_plan
        elif self.score_plan is not None:
            self.score_plan.update(
                {
                    "templates_per_device": padded_templates_per_device,
                    "template_count": int(templates.shape[0]),
                    "batches_per_device": padded_templates_per_device // batch_size,
                    "subvolume_count": self.processor.subvolume_count,
                }
            )

        LOGGER.info(
            "KLT score model: templates=%d | devices=%d | local padded=%d | "
            "FFT batch=%d | batches/device/subvolume=%d | subvolumes=%d",
            templates.shape[0],
            device_count,
            padded_templates_per_device,
            batch_size,
            padded_templates_per_device // batch_size,
            self.processor.subvolume_count,
        )
        if self.score_plan is not None and "estimated_peak_bytes" in self.score_plan:
            LOGGER.info(
                "KLT score memory estimate/device: peak=%.2f GiB | "
                "budget=%.2f GiB | fixed=%.2f GiB",
                int(self.score_plan["estimated_peak_bytes"]) / 2**30,
                int(self.score_plan["budget_bytes"]) / 2**30,
                int(self.score_plan["fixed_bytes"]) / 2**30,
            )

        template_shards = []
        normalization_shards = []
        weight_shards = []
        for device_index in range(device_count):
            start = device_index * templates_per_device
            stop = min(start + templates_per_device, templates.shape[0])
            count = max(0, stop - start)
            template_shard = np.zeros(
                (padded_templates_per_device, *templates.shape[1:]),
                dtype=np.complex64,
            )
            normalization_shard = np.zeros(
                padded_templates_per_device,
                dtype=np.float32,
            )
            weight_shard = np.zeros(
                padded_templates_per_device,
                dtype=np.float32,
            )
            if count:
                template_shard[:count] = np.asarray(
                    templates[start:stop],
                    dtype=np.complex64,
                )
                normalization_shard[:count] = normalization[start:stop]
                weight_shard[:count] = score_weights[start:stop]
            template_shards.append(template_shard)
            normalization_shards.append(normalization_shard)
            weight_shards.append(weight_shard)

        LOGGER.info("Transferring compact template shards to devices")
        device_templates = jax.device_put_sharded(template_shards, self.devices)
        device_normalization = jax.device_put_sharded(
            normalization_shards,
            self.devices,
        )
        device_weights = jax.device_put_sharded(weight_shards, self.devices)
        del template_shards, normalization_shards, weight_shards
        configured_score = partial(
            score_template_shards_and_extract_candidates,
            core_shape=self.processor.core_shape,
            source_shape=self.source.shape,
            whitening_radius=whitening_radius,
            template_radius=template_radius,
            candidate_capacity=candidate_capacity,
            template_batch_size=batch_size,
        )
        distributed_score = jax.pmap(
            configured_score,
            axis_name=_TEMPLATE_AXIS_NAME,
            in_axes=(None, None, None, 0, 0, 0, None),
            devices=self.devices,
        )
        LOGGER.info(
            "Scoring kernel will compile on the first of %d subvolumes",
            self.processor.subvolume_count,
        )

        candidate_blocks = []
        total_local_maxima = 0
        retained_local_maxima = 0
        truncated_subvolumes = 0
        valid_lower = np.full(3, template_radius)
        valid_upper = np.asarray(self.source.shape) - template_radius
        for region in tqdm(
            self.processor.regions(),
            total=self.processor.subvolume_count,
            desc="Template-sharded KLT scoring",
            unit="subvolume",
        ):
            loaded_subvolume = self.processor.load_region(region, total_halo)
            output = distributed_score(
                loaded_subvolume,
                np.asarray(region.start, dtype=np.int32),
                whitening_filter,
                device_templates,
                device_normalization,
                device_weights,
                score_offset,
            )
            local_coordinates, local_scores, local_counts = (
                np.asarray(output_leaf[0]) for output_leaf in output
            )
            candidate_count = int(local_counts)
            retained = min(candidate_count, candidate_capacity)
            total_local_maxima += candidate_count
            retained_local_maxima += retained
            truncated_subvolumes += int(candidate_count > candidate_capacity)
            coordinates = (
                np.asarray(local_coordinates[:retained])
                + np.asarray(region.start)
            )
            scores = np.asarray(local_scores[:retained])
            inside_source = np.all(
                (coordinates >= valid_lower) & (coordinates < valid_upper),
                axis=1,
            )
            if np.any(inside_source):
                candidate_blocks.append(
                    np.column_stack(
                        (coordinates[inside_source], scores[inside_source])
                    )
                )
        if not candidate_blocks:
            return np.empty((0, 4), dtype=np.float64)
        LOGGER.info(
            "Local-max retention: retained=%d / found=%d | top-K=%d | "
            "truncated subvolumes=%d/%d",
            retained_local_maxima,
            total_local_maxima,
            candidate_capacity,
            truncated_subvolumes,
            self.processor.subvolume_count,
        )
        return np.asarray(np.concatenate(candidate_blocks), dtype=np.float64)

    def non_maximum_suppression(
        self,
        candidates: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        """Apply deterministic global ranking, thresholding, and spherical NMS."""
        candidates = np.asarray(candidates, dtype=np.float64)
        particle_limit = (
            candidates.shape[0]
            if self.model.num_particles == -1
            else self.model.num_particles
        )
        ranked = ranked_candidate_nms_3d(
            candidates,
            radius=self.model.nms_radius_voxels,
            max_picks=particle_limit,
        )
        if ranked.size == 0:
            return ranked
        maximum_score = float(np.max(candidates[:, 3]))
        epsilon = np.finfo(np.float64).eps
        score_scale = (
            maximum_score
            if abs(maximum_score) > epsilon
            else np.copysign(epsilon, maximum_score or 1.0)
        )
        ranked[:, 3] /= score_scale
        passes_threshold = ranked[:, 3] > self.model.threshold
        if np.all(passes_threshold):
            return ranked
        return ranked[: int(np.argmax(~passes_threshold))]

    def process_tomogram(self) -> tuple[int, npt.NDArray[np.float64]]:
        """Run both RPSD iterations, templating, scoring, and global NMS."""
        self.bandpass_filter = self.build_bandpass_filter()
        self.initial_rpsds = self.estimate_rpsds(
            self.bandpass_filter,
            description="Band-passed RPSD extraction",
        )
        self.initial_model = self.fit_rpsds(self.initial_rpsds)
        self.whitening_filter = self.build_whitening_filter(
            self.initial_model.noise_psd
        )
        self.whitened_rpsds = self.estimate_rpsds(
            self.whitening_filter,
            description="Band-passed whitened RPSD extraction",
        )
        self.whitened_model = self.fit_rpsds(self.whitened_rpsds)
        self.templates = self.build_templates(self.whitened_model.particle_psd)
        self.prepare_score_filters(
            self.templates,
            self.whitened_model.noise_variance,
        )
        self.candidates = self.score_candidates(
            self.templates,
            self.whitened_model.noise_variance,
            self.whitening_filter,
        )
        self.particles = self.non_maximum_suppression(self.candidates)
        return self.particles.shape[0], self.particles
