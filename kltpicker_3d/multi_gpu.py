"""Single-host multi-GPU orchestration for three-dimensional KLT detection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

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
    construct_finite_whitening_filter,
)

_ALS_CONVERGENCE_TOLERANCE = 1e-4
_NOISE_PATCH_FRACTION = 0.25
_LOCAL_MAXIMUM_RADIUS = 1


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


def construct_klt_score_filters(
    templates: npt.ArrayLike,
    template_eigenvalues: npt.ArrayLike,
    noise_variance: float,
) -> tuple[
    npt.NDArray[np.generic],
    npt.NDArray[np.float32],
    np.float32,
]:
    """Convert KLT templates into convolution kernels and score weights."""
    templates = np.asarray(templates)
    eigenvalues = np.asarray(template_eigenvalues)
    if templates.ndim != 4 or templates.shape[0] < 1:
        raise ValueError("templates must have shape (modes, z, y, x)")
    if len(set(templates.shape[1:])) != 1 or templates.shape[1] % 2 == 0:
        raise ValueError("templates must have an odd cubic spatial shape")
    if eigenvalues.shape != (templates.shape[0],):
        raise ValueError("each template must have one eigenvalue")
    if noise_variance <= 0:
        raise ValueError("noise_variance must be positive")

    flattened = jnp.asarray(templates).reshape(templates.shape[0], -1)
    eigenvalues_jax = jnp.asarray(eigenvalues)
    orthonormal_basis, triangular_factor = jnp.linalg.qr(
        flattened.T,
        mode="reduced",
    )
    covariance = (
        triangular_factor * eigenvalues_jax[None, :]
    ) @ jnp.conj(triangular_factor.T) + noise_variance * jnp.eye(
        triangular_factor.shape[0],
        dtype=triangular_factor.dtype,
    )
    covariance_values, covariance_vectors = jnp.linalg.eigh(covariance)
    weights = 1.0 / noise_variance - 1.0 / covariance_values
    offset = jnp.linalg.slogdet(covariance / noise_variance)[1]
    score_basis = orthonormal_basis @ covariance_vectors[:, ::-1]
    kernels = score_basis.T.reshape((-1, *templates.shape[1:]))
    return (
        np.asarray(kernels),
        np.asarray(weights[::-1], dtype=np.float32),
        np.float32(offset),
    )


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
        devices: Sequence[jax.Device] | None = None,
        core_patch_shape: tuple[int, int, int] | None = None,
        device_memory_bytes: int | None = None,
        memory_fraction: float = 0.3,
        resident_volume_copies: int = 8,
        patches_per_microbatch: int = 1,
        candidate_capacity_per_subvolume: int = 10000,
        legendre_order: int = 150,
        threshold: float = 0,
        max_iter: int = 500,
        max_order: int = 10,
        psd_patch_size: int | None = None,
        fredholm_radius: float | None = None,
        template_side: int | None = None,
        nms_radius: float | None = None,
        boundary_mode: PaddingMode = "constant",
    ) -> None:
        """Initialize shared streaming geometry and the KLT template model."""
        selected_devices = tuple(jax.devices() if devices is None else devices)
        if not selected_devices:
            raise ValueError("at least one JAX device is required")
        if whitening_support_radius < 0:
            raise ValueError("whitening_support_radius must be nonnegative")
        if patches_per_microbatch < 1:
            raise ValueError("patches_per_microbatch must be positive")
        if candidate_capacity_per_subvolume < 1:
            raise ValueError("candidate capacity must be positive")

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
        if core_patch_shape is None:
            memory_limit = (
                device_memory_bytes
                if device_memory_bytes is not None
                else estimate_device_memory_limit(selected_devices)
            )
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
        self.candidate_capacity_per_subvolume = candidate_capacity_per_subvolume
        self.processor = MultiGPUSubvolumeProcessor(
            source,
            domain_shape=source.shape,
            core_shape=tuple(size * patch_size for size in core_patch_shape),
            devices=selected_devices,
            boundary_mode=boundary_mode,
        )

        self.initial_rpsds: RpsdExtractionResult | None = None
        self.initial_model: CalibratedRpsdModel | None = None
        self.whitening_filter: npt.NDArray[np.float32] | None = None
        self.whitened_rpsds: RpsdExtractionResult | None = None
        self.whitened_model: CalibratedRpsdModel | None = None
        self.templates: npt.NDArray[np.generic] | None = None
        self.candidates: npt.NDArray[np.float64] | None = None
        self.particles: npt.NDArray[np.float64] | None = None

    def estimate_rpsds(
        self,
        whitening_filter: npt.ArrayLike | None = None,
    ) -> RpsdExtractionResult:
        """Run streamed RPSD extraction, optionally with finite whitening."""
        return extract_streamed_rpsds(
            self.processor,
            self.patch_size,
            patches_per_microbatch=self.patches_per_microbatch,
            spatial_filter=whitening_filter,
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

    def score_candidates(
        self,
        templates: npt.ArrayLike,
        noise_variance: float,
        whitening_filter: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        """Stream fused whitening/scoring and return globally located maxima."""
        if self.model.eigvals is None:
            raise RuntimeError("template eigenvalues have not been initialized")
        whitening_filter = np.asarray(whitening_filter, dtype=np.float32)
        whitening_radius = spatial_filter_radius(whitening_filter)
        score_kernels, score_weights, score_offset = construct_klt_score_filters(
            templates,
            self.model.eigvals,
            noise_variance,
        )
        template_radius = score_kernels.shape[1] // 2
        candidate_capacity = min(
            self.candidate_capacity_per_subvolume,
            int(np.prod(self.processor.core_shape)),
        )
        total_halo = (
            whitening_radius + template_radius + _LOCAL_MAXIMUM_RADIUS
        )
        outputs = self.processor.map(
            score_subvolume_candidates,
            whitening_filter,
            score_kernels,
            score_weights,
            score_offset,
            halo=total_halo,
            pass_region_starts=True,
            static_kwargs={
                "core_shape": self.processor.core_shape,
                "source_shape": self.source.shape,
                "whitening_radius": whitening_radius,
                "template_radius": template_radius,
                "candidate_capacity": candidate_capacity,
            },
            description="KLT scoring",
        )

        candidate_blocks = []
        valid_lower = np.full(3, template_radius)
        valid_upper = np.asarray(self.source.shape) - template_radius
        for regions, output in outputs:
            local_coordinates, local_scores, local_counts = output
            for slot, region in enumerate(regions):
                if region is None:
                    continue
                candidate_count = int(local_counts[slot])
                if candidate_count > candidate_capacity:
                    raise RuntimeError(
                        "candidate capacity exceeded in subvolume starting at "
                        f"{region.start}: found {candidate_count}, capacity "
                        f"{candidate_capacity}; increase "
                        "candidate_capacity_per_subvolume"
                    )
                retained = min(candidate_count, candidate_capacity)
                coordinates = (
                    np.asarray(local_coordinates[slot, :retained])
                    + np.asarray(region.start)
                )
                scores = np.asarray(local_scores[slot, :retained])
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
        self.initial_rpsds = self.estimate_rpsds()
        self.initial_model = self.fit_rpsds(self.initial_rpsds)
        self.whitening_filter = self.build_whitening_filter(
            self.initial_model.noise_psd
        )
        self.whitened_rpsds = self.estimate_rpsds(self.whitening_filter)
        self.whitened_model = self.fit_rpsds(self.whitened_rpsds)
        self.templates = self.build_templates(self.whitened_model.particle_psd)
        self.candidates = self.score_candidates(
            self.templates,
            self.whitened_model.noise_variance,
            self.whitening_filter,
        )
        self.particles = self.non_maximum_suppression(self.candidates)
        return self.particles.shape[0], self.particles
