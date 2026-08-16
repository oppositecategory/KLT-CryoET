"""End-to-end three-dimensional KLT particle detection."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from scipy.special import roots_legendre, spherical_jn

from kltpicker_3d.alt_least_squares import (
    alternating_least_squares_solver,
)
from kltpicker_3d.fredholm_solver import (
    solve_radial_fredholm_equation,
)
from kltpicker_3d.spectral_estimation import (
    estimate_isotropic_powerspectrum_tensor,
)
from kltpicker_3d.utils import (
    bandpass_filter_3d,
    calibrate_radial_psds,
    expand_spherical_harmonic_templates,
    generate_uniform_radial_sampling_points,
    prewhiten_tomogram,
    radial_average_jax,
    radial_mode_truncation_index,
    ranked_local_maxima_nms_3d,
    trigonometric_interpolation,
)

_ALS_CONVERGENCE_TOLERANCE = 1e-4
_NOISE_PATCH_FRACTION = 0.25
_PSD_ACF_DISTANCE_FRACTION = 0.3
_TEMPLATE_ENERGY_FRACTION = 0.99

_vectorized_spectrum_estimation = jax.vmap(
    estimate_isotropic_powerspectrum_tensor,
    in_axes=(0, None),
)
_vectorized_radial_average = jax.vmap(
    radial_average_jax,
    in_axes=(0, None, None, None),
)


def _odd_floor(value: float) -> int:
    """Return the largest odd integer not greater than ``value``."""
    size = int(np.floor(value))
    return size if size % 2 else size - 1


def _extract_nonoverlapping_patches(
    volume: jax.Array | npt.ArrayLike,
    patch_size: int,
) -> jax.Array:
    """Extract every complete cubic patch independently along each axis.

    Args:
        volume: Three-dimensional tomogram in ``(z, y, x)`` order.
        patch_size: Cubic patch side length, in voxels.

    Returns:
        JAX array with shape
        ``(num_patches, patch_size, patch_size, patch_size)``.

    Raises:
        ValueError: If ``volume`` is not 3D or any axis cannot fit one patch.
    """
    volume_array = jnp.asarray(volume)
    if volume_array.ndim != 3:
        raise ValueError("volume must be three-dimensional")
    if patch_size < 1:
        raise ValueError("patch_size must be positive")

    blocks = np.asarray(volume_array.shape, dtype=np.int64) // patch_size
    if np.any(blocks < 1):
        raise ValueError(
            "all tomogram dimensions must be at least as large as patch_size"
        )

    cropped_shape = blocks * patch_size
    cropped = volume_array[
        : cropped_shape[0],
        : cropped_shape[1],
        : cropped_shape[2],
    ]
    blocks_z, blocks_y, blocks_x = (int(value) for value in blocks)
    patches = cropped.reshape(
        blocks_z,
        patch_size,
        blocks_y,
        patch_size,
        blocks_x,
        patch_size,
    ).transpose(0, 2, 4, 1, 3, 5)
    return patches.reshape(
        blocks_z * blocks_y * blocks_x,
        patch_size,
        patch_size,
        patch_size,
    )


class KLTParticleDetector3D:
    """Detect particles using data-adaptive three-dimensional KLT templates.

    ``particle_diameter`` may be expressed in physical units.
    ``mgscale`` converts those units to voxels. All internal geometric
    calculations use the derived voxel quantities.
    """

    def __init__(
        self,
        tomogram: jax.Array | npt.ArrayLike | None,
        particle_diameter: float,
        mgscale: float,
        num_particles: int,
        legendre_order: int = 150,
        threshold: float = 0,
        max_iter: int = 500,
        max_order: int = 10,
        template_energy_fraction: float = _TEMPLATE_ENERGY_FRACTION,
        max_templates: int | None = None,
        psd_patch_size: int | None = None,
        fredholm_radius: float | None = None,
        template_side: int | None = None,
        bandpass_low_fraction: float = 0.05,
        bandpass_high_fraction: float = 0.05,
        nms_radius: float | None = None,
    ) -> None:
        """Initialize detector geometry and numerical settings.

        Args:
            tomogram: Optional input volume in ``(z, y, x)`` order. ``None``
                initializes only the reusable KLT geometry and template model.
            particle_diameter: Particle diameter in input physical units.
            mgscale: Voxels per input physical unit.
            num_particles: Maximum requested picks, or ``-1`` to use only the
                iteration limit and score threshold.
            legendre_order: Gauss-Legendre quadrature order.
            threshold: Minimum normalized score for a retained peak.
            max_iter: Maximum ALS iterations and legacy maximum unconstrained
                pick count.
            max_order: Number of spherical-harmonic orders, starting at zero.
            template_energy_fraction: Fraction of the degeneracy-weighted KLT
                eigenvalue energy retained when constructing templates.
            max_templates: Optional upper bound on complete ``(ell, n, m)``
                templates. Multiplets are never split to reach the bound.
            psd_patch_size: Optional odd PSD patch side, in voxels.
            fredholm_radius: Optional Fredholm support radius, in voxels.
            template_side: Optional odd template side, in voxels.
            bandpass_low_fraction: Fraction of low radial frequencies removed.
            bandpass_high_fraction: Fraction of high radial frequencies
                removed.
            nms_radius: Optional NMS center-distance radius, in voxels.

        Raises:
            ValueError: If the volume, geometry, or numerical settings are
                invalid.
        """
        tomogram_array = None
        if tomogram is not None:
            tomogram_array = jnp.asarray(
                tomogram,
                dtype=jnp.result_type(tomogram, jnp.float32),
            )
            if tomogram_array.ndim != 3:
                raise ValueError("tomogram must be three-dimensional")
        if particle_diameter <= 0:
            raise ValueError("particle_diameter must be positive")
        if mgscale <= 0:
            raise ValueError("mgscale must be positive")
        if num_particles < -1:
            raise ValueError("num_particles must be -1 or nonnegative")
        if legendre_order < 1:
            raise ValueError("legendre_order must be positive")
        if max_iter < 1:
            raise ValueError("max_iter must be positive")
        if max_order < 1:
            raise ValueError("max_order must be positive")
        if not 0 < template_energy_fraction <= 1:
            raise ValueError("template_energy_fraction must lie in (0, 1]")
        if max_templates is not None and max_templates < 1:
            raise ValueError("max_templates must be positive when provided")

        self.tomogram = tomogram_array
        self.particle_diameter = particle_diameter
        self.mgscale = mgscale
        self.max_order = max_order
        self.template_energy_fraction = float(template_energy_fraction)
        self.max_templates = max_templates
        self.bandlimit = np.pi
        self.bandpass_low_fraction = bandpass_low_fraction
        self.bandpass_high_fraction = bandpass_high_fraction

        scaled_diameter = self.mgscale * self.particle_diameter
        self.particle_diameter_voxels = int(np.floor(scaled_diameter))
        self.nms_radius_voxels = (
            0.5 * scaled_diameter if nms_radius is None else nms_radius
        )
        if self.nms_radius_voxels < 0:
            raise ValueError("nms_radius must be nonnegative")

        default_patch_size = _odd_floor(0.8 * scaled_diameter)
        self.patch_size = (
            default_patch_size if psd_patch_size is None else psd_patch_size
        )
        self.fredholm_radius_voxels = (
            0.4 * scaled_diameter if fredholm_radius is None else fredholm_radius
        )
        default_template_side = 2 * int(np.ceil(self.fredholm_radius_voxels)) + 1
        self.template_side = (
            default_template_side if template_side is None else template_side
        )

        if self.patch_size < 3 or self.patch_size % 2 == 0:
            raise ValueError("psd_patch_size must be an odd integer at least 3")
        if self.fredholm_radius_voxels <= 0:
            raise ValueError("fredholm_radius must be positive")
        if self.template_side < 3 or self.template_side % 2 == 0:
            raise ValueError("template_side must be an odd integer at least 3")
        template_grid_radius = (self.template_side - 1) / 2
        if template_grid_radius < self.fredholm_radius_voxels:
            raise ValueError(
                "template_side is too small to contain the Fredholm support"
            )
        if self.particle_diameter_voxels < 3:
            raise ValueError(
                "particle_diameter * mgscale must span at least three voxels"
            )

        # These legacy names are retained because experiment notebooks use
        # them as array geometry rather than as independent model parameters.
        self.template_diameter = self.template_side
        self.template_radius_voxels = self.fredholm_radius_voxels
        self.available_radial_mode_count: int | None = None
        self.available_template_count: int | None = None
        self.retained_radial_mode_count: int | None = None
        self.retained_template_count: int | None = None
        self.retained_template_energy_fraction: float | None = None
        self.max_iter = max_iter

        spectrum_size = 2 * self.patch_size - 1
        (
            self.uniform_points,
            self.shell_ids,
            self.counts,
        ) = generate_uniform_radial_sampling_points(
            spectrum_size,
            self.bandlimit,
        )

        self.legendre_order = legendre_order
        self.num_particles = num_particles
        self.threshold = threshold

        self.eigvals: npt.NDArray[np.generic] | None = None
        self.eigfuncs: npt.NDArray[np.generic] | None = None
        self.score_mat: npt.NDArray[np.float64] | None = None
        self.preprocessed_tomogram: jax.Array | None = None
        self.whitened_tomogram: jax.Array | None = None
        self.initial_particle_psd: npt.NDArray[np.float64] | None = None
        self.initial_noise_psd: npt.NDArray[np.float64] | None = None
        self.particle_psd: npt.NDArray[np.float64] | None = None
        self.noise_psd: npt.NDArray[np.float64] | None = None
        self.noise_variance: float | None = None
        self.radial_eigvals: npt.NDArray[np.float64] | None = None
        self.template_orders: npt.NDArray[np.int64] | None = None
        self.template_m_values: npt.NDArray[np.int64] | None = None
        self.templates: npt.NDArray[np.generic] | None = None

    def process_tomogram(
        self,
    ) -> tuple[int, npt.NDArray[np.float64]]:
        """Run preprocessing, spectral estimation, templating, and picking."""
        if self.tomogram is None:
            raise RuntimeError("process_tomogram requires a resident tomogram")
        self.preprocessed_tomogram = bandpass_filter_3d(
            self.tomogram,
            low_fraction=self.bandpass_low_fraction,
            high_fraction=self.bandpass_high_fraction,
            normalize=True,
        )

        # The first estimate supplies the noise model used to whiten the full
        # tomogram. The second estimate and scoring both use that same whitened
        # representation.
        (
            initial_particle_psd,
            initial_noise_psd,
            _,
        ) = self.factorize_rpsd(self.preprocessed_tomogram)
        self.initial_particle_psd = initial_particle_psd
        self.initial_noise_psd = initial_noise_psd
        self.whitened_tomogram = prewhiten_tomogram(
            self.preprocessed_tomogram,
            self.uniform_points,
            initial_noise_psd,
        )

        (
            particle_psd,
            noise_psd,
            noise_variance,
        ) = self.factorize_rpsd(self.whitened_tomogram)
        self.particle_psd = particle_psd
        self.noise_psd = noise_psd
        self.noise_variance = noise_variance

        (
            radial_eigenvalues,
            radial_eigenfunctions,
            angular_orders,
            particle_psd_nodes,
        ) = self._solve_radial_modes(particle_psd)
        templates, _ = self.create_gpsf_templates(
            radial_eigenvalues,
            radial_eigenfunctions,
            angular_orders,
            particle_psd_nodes,
        )
        self.templates = templates
        return self.detect_particles(
            templates,
            noise_variance,
            self.whitened_tomogram,
        )

    def _solve_radial_modes(
        self,
        particle_psd: npt.ArrayLike,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.generic],
        npt.NDArray[np.int64],
        npt.NDArray[np.float64],
    ]:
        """Interpolate the particle PSD and solve every angular order."""
        legendre_nodes, _ = roots_legendre(self.legendre_order)
        frequency_nodes = (self.bandlimit / 2) * (legendre_nodes + 1)
        particle_psd_nodes = np.maximum(
            np.asarray(
                trigonometric_interpolation(
                    self.uniform_points,
                    particle_psd,
                    frequency_nodes,
                )
            ),
            0,
        )

        eigenvalue_blocks = []
        eigenfunction_blocks = []
        for angular_order in range(self.max_order):
            eigenvalues, eigenfunctions, _ = solve_radial_fredholm_equation(
                particle_psd_nodes,
                angular_order,
                self.fredholm_radius_voxels,
                self.bandlimit,
                K=self.legendre_order,
            )
            eigenvalue_blocks.append(eigenvalues)
            eigenfunction_blocks.append(eigenfunctions)

        eigenvalues = np.asarray(eigenvalue_blocks).reshape(-1)
        # Each solver matrix stores eigenfunctions in columns. Move the radial
        # mode axis before flattening so global sorting preserves pairing.
        eigenfunctions = (
            np.asarray(eigenfunction_blocks)
            .transpose(0, 2, 1)
            .reshape(-1, self.legendre_order)
        )
        angular_orders = np.repeat(
            np.arange(self.max_order, dtype=np.int64),
            self.legendre_order,
        )

        descending = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[descending]
        eigenfunctions = eigenfunctions[descending]
        angular_orders = angular_orders[descending]

        active = eigenvalues > np.spacing(1)
        if not np.any(active):
            raise ValueError("Fredholm solve returned no positive eigenvalues")
        return (
            eigenvalues[active],
            eigenfunctions[active],
            angular_orders[active],
            particle_psd_nodes,
        )

    def factorize_rpsd(
        self,
        tomogram: jax.Array | npt.ArrayLike | None = None,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        float,
    ]:
        """Estimate and variance-calibrate particle and noise radial PSDs.

        Args:
            tomogram: Optional volume to factorize. Defaults to the original
                detector input.

        Returns:
            Calibrated particle RPSD, calibrated noise RPSD, and the robust
            spatial noise-variance estimate.
        """
        source = self.tomogram if tomogram is None else tomogram
        if source is None:
            raise RuntimeError("factorize_rpsd requires a resident tomogram")
        max_distance = int(np.floor(_PSD_ACF_DISTANCE_FRACTION * self.patch_size))
        patches = _extract_nonoverlapping_patches(
            source,
            self.patch_size,
        )
        patches = patches - jnp.mean(
            patches,
            axis=(1, 2, 3),
            keepdims=True,
        )

        patch_variances = jnp.var(patches, axis=(1, 2, 3))
        sorted_patch_variances = jnp.sort(patch_variances)
        noise_patch_count = max(
            1,
            int(np.floor(_NOISE_PATCH_FRACTION * patch_variances.size)),
        )
        noise_variance = jnp.mean(sorted_patch_variances[:noise_patch_count])
        mean_patch_variance = jnp.mean(patch_variances)

        power_spectra = _vectorized_spectrum_estimation(
            patches,
            max_distance,
        )
        radial_spectra = _vectorized_radial_average(
            power_spectra,
            self.shell_ids,
            self.counts,
            self.uniform_points.shape[0],
        )
        factorization = alternating_least_squares_solver(
            radial_spectra,
            self.max_iter,
            _ALS_CONVERGENCE_TOLERANCE,
        )
        particle_psd, noise_psd = calibrate_radial_psds(
            self.uniform_points,
            factorization.gamma,
            factorization.v,
            float(noise_variance),
            float(mean_patch_variance),
        )
        return particle_psd, noise_psd, float(noise_variance)

    def factorize_RPSD(  # noqa: N802
        self,
        tomogram: jax.Array | npt.ArrayLike | None = None,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        float,
    ]:
        """Call :meth:`factorize_rpsd` for backward compatibility."""
        return self.factorize_rpsd(tomogram)

    def create_gpsf_templates(
        self,
        eigenvalues: npt.ArrayLike,
        eigenfunctions: npt.ArrayLike,
        angular_orders: npt.ArrayLike,
        particle_psd_nodes: npt.ArrayLike,
    ) -> tuple[
        npt.NDArray[np.generic],
        npt.NDArray[np.generic],
    ]:
        """Generate complete generalized prolate spheroidal template modes.

        Args:
            eigenvalues: Descending positive radial KLT eigenvalues.
            eigenfunctions: Radial eigenfunctions sampled at spatial
                quadrature nodes.
            angular_orders: Spherical-harmonic order of each radial mode.
            particle_psd_nodes: Particle PSD at frequency quadrature nodes.

        Returns:
            Complete spherical-harmonic templates and one eigenvalue per
            returned angular mode.
        """
        eigenvalues = np.asarray(eigenvalues)
        eigenfunctions = np.asarray(eigenfunctions)
        angular_orders = np.asarray(
            angular_orders,
            dtype=np.int64,
        )
        particle_psd_nodes = np.asarray(particle_psd_nodes)

        support_radius = self.fredholm_radius_voxels
        bandlimit = self.bandlimit
        quadrature_order = self.legendre_order

        grid_radius = (self.template_side - 1) // 2
        grid_axis = np.arange(-grid_radius, grid_radius + 1)
        grid_z, grid_y, grid_x = np.meshgrid(
            grid_axis,
            grid_axis,
            grid_axis,
            indexing="ij",
        )
        radius_tensor = np.sqrt(grid_z**2 + grid_y**2 + grid_x**2)
        support_mask = radius_tensor <= support_radius
        uniform_radii, inverse_radius_indices = np.unique(
            radius_tensor[support_mask],
            return_inverse=True,
        )

        legendre_nodes, legendre_weights = roots_legendre(quadrature_order)
        spatial_nodes = (support_radius / 2) * (legendre_nodes + 1)
        frequency_nodes = (bandlimit / 2) * (legendre_nodes + 1)

        multiplicities = 2 * angular_orders + 1
        mode_energy = eigenvalues * multiplicities
        total_energy = float(np.sum(mode_energy))
        self.available_radial_mode_count = int(eigenvalues.size)
        self.available_template_count = int(np.sum(multiplicities))

        energy_truncation_index = radial_mode_truncation_index(
            eigenvalues,
            angular_orders,
            energy_fraction=self.template_energy_fraction,
        )
        truncation_index = energy_truncation_index
        if self.max_templates is not None:
            complete_template_counts = np.cumsum(multiplicities)
            cap_truncation_index = int(
                np.searchsorted(
                    complete_template_counts,
                    self.max_templates,
                    side="right",
                )
            )
            if cap_truncation_index < 1:
                raise ValueError(
                    "max_templates is too small to retain the leading "
                    "spherical-harmonic multiplet"
                )
            truncation_index = min(
                truncation_index,
                cap_truncation_index,
            )
        self.retained_radial_mode_count = truncation_index
        self.retained_template_count = int(
            np.sum(multiplicities[:truncation_index])
        )
        self.retained_template_energy_fraction = float(
            np.sum(mode_energy[:truncation_index]) / total_energy
        )
        eigenfunctions = eigenfunctions[:truncation_index]
        eigenvalues = eigenvalues[:truncation_index]
        angular_orders = angular_orders[:truncation_index]

        self.eigfuncs = eigenfunctions
        self.radial_eigvals = eigenvalues

        uniform_grid = np.outer(
            uniform_radii,
            frequency_nodes,
        )
        quadrature_grid = np.outer(
            spatial_nodes,
            frequency_nodes,
        )

        def radial_basis(
            values: npt.ArrayLike,
            order: int,
        ) -> npt.NDArray[np.complex128]:
            values_array = np.asarray(values)
            return np.asarray(
                4 * np.pi * (1j**order) * spherical_jn(order, values_array),
                dtype=np.complex128,
            )

        maximum_order = int(angular_orders.max()) + 1
        uniform_bases = np.asarray(
            [radial_basis(uniform_grid, order) for order in range(maximum_order)]
        )[angular_orders]
        quadrature_bases = np.asarray(
            [radial_basis(quadrature_grid, order) for order in range(maximum_order)]
        )[angular_orders]

        parity_sign = np.where(
            angular_orders % 2,
            -1,
            1,
        )
        frequency_weights = (
            bandlimit / 2 * legendre_weights * particle_psd_nodes * frequency_nodes**2
        )
        spatial_weights = support_radius / 2 * legendre_weights * spatial_nodes**2
        right_basis = parity_sign[:, None, None] * quadrature_bases
        interpolation_operator = (
            uniform_bases * frequency_weights[None, None, :]
        ) @ right_basis
        uniform_eigenfunctions = (
            np.einsum(
                "bik,k,bk->bi",
                interpolation_operator,
                spatial_weights,
                eigenfunctions,
                optimize=True,
            )
            / eigenvalues[:, None]
        )

        template_count = int(np.sum(2 * angular_orders + 1))
        templates = np.empty(
            (template_count, *radius_tensor.shape),
            dtype=np.complex64,
        )
        template_eigenvalues = np.empty(template_count, dtype=np.float64)
        template_orders = np.empty(template_count, dtype=np.int64)
        template_m_values = np.empty(template_count, dtype=np.int64)

        angular_cache: dict[int, npt.NDArray[np.complex64]] = {}
        output_start = 0
        for radial_index, angular_order in enumerate(angular_orders):
            order = int(angular_order)
            if order not in angular_cache:
                angular_modes, _, _ = (
                    expand_spherical_harmonic_templates(
                        np.ones((1, *radius_tensor.shape), dtype=np.float32),
                        np.asarray([order]),
                        grid_x,
                        grid_y,
                        grid_z,
                    )
                )
                angular_cache[order] = np.asarray(
                    angular_modes,
                    dtype=np.complex64,
                )
            angular_modes = angular_cache[order]
            multiplicity = angular_modes.shape[0]
            output_stop = output_start + multiplicity
            radial_template = np.zeros(radius_tensor.shape, dtype=np.complex64)
            radial_template[support_mask] = np.asarray(
                uniform_eigenfunctions[
                    radial_index,
                    inverse_radius_indices,
                ],
                dtype=np.complex64,
            )
            templates[output_start:output_stop] = (
                angular_modes * radial_template[None, ...]
            )
            template_eigenvalues[output_start:output_stop] = eigenvalues[
                radial_index
            ]
            template_orders[output_start:output_stop] = order
            template_m_values[output_start:output_stop] = np.arange(
                -order,
                order + 1,
                dtype=np.int64,
            )
            output_start = output_stop

        self.eigvals = template_eigenvalues
        self.template_orders = template_orders
        self.template_m_values = template_m_values
        return templates, template_eigenvalues

    def create_GPSF_templates(  # noqa: N802
        self,
        eigvals: npt.ArrayLike,
        eigfuncs: npt.ArrayLike,
        orders: npt.ArrayLike,
        G: npt.ArrayLike,
    ) -> tuple[
        npt.NDArray[np.generic],
        npt.NDArray[np.generic],
    ]:
        """Call :meth:`create_gpsf_templates` for compatibility."""
        return self.create_gpsf_templates(
            eigvals,
            eigfuncs,
            orders,
            G,
        )

    def detect_particles(
        self,
        templates: npt.ArrayLike,
        noise_var_approx: float,
        tomogram: jax.Array | npt.ArrayLike | None = None,
    ) -> tuple[int, npt.NDArray[np.float64]]:
        """Score the tomogram with KLT templates and pick local maxima.

        Args:
            templates: KLT templates with shape
                ``(num_templates, z, y, x)``.
            noise_var_approx: Positive spatial noise variance.
            tomogram: Optional scoring volume. Defaults to the original input.

        Returns:
            Number of accepted particles and an array containing
            ``(z, y, x, normalized_score)`` rows.

        Raises:
            RuntimeError: If template eigenvalues have not been initialized.
            ValueError: If template or noise inputs are invalid.
        """
        if noise_var_approx <= 0:
            raise ValueError("noise_var_approx must be positive")
        if self.eigvals is None:
            raise RuntimeError(
                "template eigenvalues must be initialized before detection"
            )

        source_volume = self.tomogram if tomogram is None else tomogram
        if source_volume is None:
            raise RuntimeError("detect_particles requires a resident tomogram")
        source = jnp.asarray(source_volume)
        template_array = jnp.asarray(templates)
        if template_array.ndim != 4:
            raise ValueError("templates must have shape (num_templates, z, y, x)")
        num_templates, size_z, size_y, size_x = template_array.shape
        flattened_templates = template_array.reshape(
            num_templates,
            size_z * size_y * size_x,
        )
        template_eigenvalues = jnp.asarray(self.eigvals)
        if template_eigenvalues.size != num_templates:
            raise ValueError(
                "each spherical-harmonic template must have one eigenvalue"
            )

        orthonormal_basis, triangular_factor = jnp.linalg.qr(
            flattened_templates.T,
            mode="reduced",
        )
        template_covariance = (
            triangular_factor * template_eigenvalues[None, :]
        ) @ jnp.conj(triangular_factor.T) + noise_var_approx * jnp.eye(
            triangular_factor.shape[0],
            dtype=triangular_factor.dtype,
        )
        covariance_eigenvalues, covariance_eigenvectors = jnp.linalg.eigh(
            template_covariance
        )
        score_weights = 1.0 / noise_var_approx - 1.0 / covariance_eigenvalues
        score_offset = jnp.linalg.slogdet(template_covariance / noise_var_approx)[1]

        score_weights = score_weights[::-1]
        covariance_eigenvectors = covariance_eigenvectors[:, ::-1]
        score_basis = orthonormal_basis @ covariance_eigenvectors
        kernels = score_basis.T.reshape(
            score_basis.shape[1],
            size_z,
            size_y,
            size_x,
        )

        score_shape = tuple(
            source_size - kernel_size + 1
            for source_size, kernel_size in zip(
                source.shape,
                template_array.shape[1:],
                strict=True,
            )
        )
        if any(size < 1 for size in score_shape):
            raise ValueError("templates cannot be larger than the tomogram")
        initial_score = jnp.zeros(
            score_shape,
            dtype=jnp.result_type(source, jnp.float32),
        )

        def accumulate_kernel_score(
            index: jax.Array,
            score: jax.Array,
        ) -> jax.Array:
            kernel = jnp.conj(jnp.flip(kernels[index], axis=(0, 1, 2)))
            response = jax.scipy.signal.fftconvolve(
                source,
                kernel,
                mode="valid",
            )
            return score + score_weights[index] * jnp.square(jnp.abs(response))

        score_matrix = jax.lax.fori_loop(
            0,
            kernels.shape[0],
            accumulate_kernel_score,
            initial_score,
        )
        self.score_mat = np.asarray(score_matrix - score_offset)
        return self.picking_from_scoring_vol_3d(self.score_mat)

    def picking_from_scoring_vol_3d(
        self,
        score_vol: npt.ArrayLike,
    ) -> tuple[int, npt.NDArray[np.float64]]:
        """Apply ranked local-maxima NMS to a detector score volume.

        Args:
            score_vol: Three-dimensional valid-convolution score volume.

        Returns:
            Number of accepted picks and rows of
            ``(z, y, x, normalized_score)``.
        """
        score_volume = np.asarray(score_vol)
        if score_volume.ndim != 3:
            raise ValueError("score_vol must be three-dimensional")
        offset = self.template_side // 2
        maximum_score = float(np.max(score_volume))
        epsilon = np.finfo(np.float64).eps
        score_scale = (
            maximum_score
            if abs(maximum_score) > epsilon
            else np.copysign(epsilon, maximum_score or 1.0)
        )

        particle_limit = np.inf if self.num_particles == -1 else self.num_particles
        pick_limit = int(min(self.max_iter, particle_limit))
        candidate_indices, candidate_values = ranked_local_maxima_nms_3d(
            score_volume,
            radius=self.nms_radius_voxels,
            max_picks=pick_limit,
        )

        particles = []
        for index, peak_score in zip(
            candidate_indices,
            candidate_values,
            strict=True,
        ):
            normalized_score = float(peak_score / score_scale)
            if normalized_score <= self.threshold:
                break
            center = index + offset
            particles.append(
                [
                    float(center[0]),
                    float(center[1]),
                    float(center[2]),
                    normalized_score,
                ]
            )

        particle_coordinates = (
            np.asarray(particles, dtype=np.float64)
            if particles
            else np.zeros((0, 4), dtype=np.float64)
        )
        return particle_coordinates.shape[0], particle_coordinates
