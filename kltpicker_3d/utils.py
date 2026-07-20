"""Numerical utilities for 3D KLT spectral estimation and detection."""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from scipy import ndimage, special


def ranked_local_maxima_nms_3d(
    score_volume: npt.ArrayLike,
    radius: float,
    max_picks: int,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.generic]]:
    """Return ranked original local maxima after spherical center-distance NMS.

    Local maxima are extracted once from the unmodified score tensor.  NMS is
    then applied only between those candidates.  This avoids creating
    artificial candidates on the boundary of previously suppressed regions.
    """
    scores = np.asarray(score_volume)
    if scores.ndim != 3:
        raise ValueError("score_volume must be three-dimensional")
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    if max_picks < 0:
        raise ValueError("max_picks must be nonnegative")
    if max_picks == 0:
        return np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=scores.dtype)

    neighborhood_max = ndimage.maximum_filter(
        scores,
        size=3,
        mode="constant",
        cval=-np.inf,
    )
    neighborhood_min = ndimage.minimum_filter(
        scores,
        size=3,
        mode="constant",
        cval=np.inf,
    )
    candidate_mask = (
        np.isfinite(scores) & (scores == neighborhood_max) & (scores > neighborhood_min)
    )
    candidate_indices = np.argwhere(candidate_mask)
    if candidate_indices.size == 0:
        return np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=scores.dtype)

    candidate_values = scores[tuple(candidate_indices.T)]
    ranked = np.argsort(-candidate_values, kind="stable")
    candidate_indices = candidate_indices[ranked]
    candidate_values = candidate_values[ranked]

    capacity = min(max_picks, candidate_indices.shape[0])
    accepted_indices = np.empty((capacity, 3), dtype=np.int64)
    accepted_values = np.empty(capacity, dtype=scores.dtype)
    accepted_count = 0
    radius_squared = radius**2
    for index, value in zip(
        candidate_indices,
        candidate_values,
        strict=True,
    ):
        if accepted_count:
            separation_squared = np.sum(
                (accepted_indices[:accepted_count] - index) ** 2,
                axis=1,
            )
            if np.any(separation_squared <= radius_squared):
                continue
        accepted_indices[accepted_count] = index
        accepted_values[accepted_count] = value
        accepted_count += 1
        if accepted_count == capacity:
            break

    return (
        accepted_indices[:accepted_count],
        accepted_values[:accepted_count],
    )


def bandpass_filter_3d(
    volume: jax.Array | npt.ArrayLike,
    low_fraction: float = 0.05,
    high_fraction: float = 0.05,
    normalize: bool = True,
) -> jax.Array:
    """Apply an isotropic 3D FFT band-pass relative to radial Nyquist.

    ``low_fraction`` and ``high_fraction`` are fractions of the radial band
    ``[0, pi]`` removed at the low- and high-frequency ends.  Frequencies in
    the FFT cube outside the retained radial band are also removed.
    """
    if not 0 <= low_fraction < 1:
        raise ValueError("low_fraction must lie in [0, 1)")
    if not 0 <= high_fraction < 1:
        raise ValueError("high_fraction must lie in [0, 1)")
    if low_fraction + high_fraction >= 1:
        raise ValueError("low_fraction + high_fraction must be less than 1")

    volume = jnp.asarray(
        volume,
        dtype=jnp.result_type(volume, jnp.float32),
    )
    if volume.ndim != 3:
        raise ValueError("volume must be three-dimensional")

    centered = volume - jnp.mean(volume)
    if normalize:
        standard_deviation = jnp.std(centered)
        centered = jnp.where(
            standard_deviation > 0,
            centered / standard_deviation,
            centered,
        )

    frequencies = [2 * jnp.pi * jnp.fft.fftfreq(size, d=1.0) for size in volume.shape]
    wz, wy, wx = jnp.meshgrid(*frequencies, indexing="ij")
    radial_frequency = jnp.sqrt(wz**2 + wy**2 + wx**2)
    low_cutoff = low_fraction * jnp.pi
    high_cutoff = (1 - high_fraction) * jnp.pi
    passband = (radial_frequency >= low_cutoff) & (radial_frequency <= high_cutoff)

    filtered = jnp.fft.ifftn(jnp.fft.fftn(centered) * passband).real
    return filtered - jnp.mean(filtered)


def radial_average_jax(
    spectrum: jax.Array,
    shell_ids: jax.Array | npt.ArrayLike,
    counts: jax.Array | npt.ArrayLike,
    nbins: int,
) -> jax.Array:
    """Average a spectrum over precomputed radial shells.

    Args:
        spectrum: Spectrum tensor to average.
        shell_ids: Integer shell index for each tensor element. Negative
            entries are outside the radial bandlimit.
        counts: Number of samples in each radial shell.
        nbins: Number of radial shells.

    Returns:
        Radial averages with shape ``(nbins,)``.
    """
    values = spectrum.ravel()
    ids = jnp.asarray(shell_ids).ravel()
    shell_counts = jnp.asarray(counts)
    mask = ids >= 0
    sums = jnp.bincount(
        ids[mask],
        weights=values[mask],
        length=nbins,
    )
    return jnp.where(
        shell_counts > 0,
        sums / shell_counts,
        0,
    )


def prewhiten_patch(
    patch: jax.Array | npt.ArrayLike,
    noise_psd: jax.Array | npt.ArrayLike,
) -> jax.Array:
    """Prewhiten a cubic patch using a centered 3D noise PSD.

    Args:
        patch: Cubic input patch.
        noise_psd: Centered cubic noise PSD at least as large as ``patch``.

    Returns:
        Prewhitened patch with the same shape as ``patch``.

    Raises:
        ValueError: If the inputs are not compatible cubic arrays.
    """
    patch = jnp.asarray(patch)
    noise_psd = jnp.asarray(noise_psd)
    if patch.ndim != 3 or len(set(patch.shape)) != 1:
        raise ValueError("patch must be a cubic 3D array")
    if noise_psd.ndim != 3 or len(set(noise_psd.shape)) != 1:
        raise ValueError("noise_psd must be a cubic 3D array")
    patch_size = patch.shape[0]
    spectrum_size = noise_psd.shape[0]
    if patch_size > spectrum_size:
        raise ValueError("noise_psd must be at least as large as patch")

    midpoint = spectrum_size // 2
    start = midpoint - patch_size // 2
    end = start + patch_size

    sqrt_noise_psd = jnp.sqrt(jnp.maximum(noise_psd, 0))
    sqrt_noise_psd = 0.5 * (sqrt_noise_psd + jnp.flip(sqrt_noise_psd, axis=(0, 1, 2)))
    spectral_norm = jnp.linalg.norm(sqrt_noise_psd)
    normalized_filter = jnp.where(
        spectral_norm > 0,
        sqrt_noise_psd / spectral_norm,
        sqrt_noise_psd,
    )
    inverse_filter = jnp.where(
        normalized_filter > jnp.finfo(normalized_filter.dtype).eps,
        jax.lax.reciprocal(normalized_filter),
        0,
    )

    padded_patch = jnp.zeros_like(noise_psd)
    padded_patch = padded_patch.at[
        start:end,
        start:end,
        start:end,
    ].set(patch)
    patch_spectrum = jnp.fft.fftshift(jnp.fft.fftn(jnp.fft.ifftshift(padded_patch)))
    whitened = jnp.fft.fftshift(
        jnp.fft.ifftn(jnp.fft.ifftshift(patch_spectrum * inverse_filter))
    ).real
    return whitened[start:end, start:end, start:end]


def radial_psd_to_variance(
    radial_points: npt.ArrayLike,
    radial_psd: npt.ArrayLike,
) -> float:
    """Integrate an isotropic 3D PSD using angular-frequency coordinates.

    With the Fourier convention used by ``numpy.fft``, the variance is the
    PSD integral over the cube ``[-pi, pi]^3``, divided by ``(2*pi)^3``.
    Radial samples cover the inscribed Nyquist sphere; consistently with
    ``prewhiten_tomogram``, the last radial value is extended through the cube
    corners.

    This convention is essential for white noise: a constant PSD of
    ``sigma**2`` must integrate to the spatial variance ``sigma**2``.
    """
    radial_points = np.asarray(radial_points, dtype=np.float64)
    radial_psd = np.asarray(radial_psd, dtype=np.float64)
    if radial_points.ndim != 1 or radial_psd.ndim != 1:
        raise ValueError("radial_points and radial_psd must be one-dimensional")
    if radial_points.shape != radial_psd.shape:
        raise ValueError("radial_points and radial_psd must have equal length")
    if radial_points.size < 2:
        raise ValueError("at least two radial samples are required")
    if np.any(np.diff(radial_points) <= 0):
        raise ValueError("radial_points must be strictly increasing")
    if radial_points[0] < 0 or radial_points[-1] > np.pi:
        raise ValueError("radial_points must lie in [0, pi]")

    integrand = radial_psd * radial_points**2
    dr = np.diff(radial_points)
    spherical_integral = 4 * np.pi * np.sum(0.5 * (integrand[:-1] + integrand[1:]) * dr)
    sampled_ball_volume = 4 * np.pi * radial_points[-1] ** 3 / 3
    cube_volume = (2 * np.pi) ** 3
    corner_integral = (cube_volume - sampled_ball_volume) * radial_psd[-1]
    return float((spherical_integral + corner_integral) / cube_volume)


def calibrate_radial_psds(
    radial_points: npt.ArrayLike,
    particle_psd: npt.ArrayLike,
    noise_psd: npt.ArrayLike,
    noise_variance: float,
    mean_patch_variance: float,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Resolve the ALS scale/offset ambiguity using measured variances.

    ALS identifies the clean spectrum only up to a positive scale and can
    absorb a multiple of that spectrum into its noise baseline.  This is the
    3D counterpart of the calibration step in the original 2D picker.
    """
    particle_psd = np.asarray(particle_psd, dtype=np.float64)
    noise_psd = np.asarray(noise_psd, dtype=np.float64)
    if noise_variance < 0 or mean_patch_variance < 0:
        raise ValueError("variance estimates must be non-negative")

    particle_shape_variance = radial_psd_to_variance(
        radial_points,
        particle_psd,
    )
    noise_shape_variance = radial_psd_to_variance(
        radial_points,
        noise_psd,
    )
    scale = max(
        1.0,
        abs(particle_shape_variance),
        abs(noise_shape_variance),
        noise_variance,
        mean_patch_variance,
    )
    eps = np.finfo(np.float64).eps * scale
    if particle_shape_variance <= eps:
        raise ValueError("ALS returned a degenerate particle spectrum")

    # If ALS returned v_hat = v + b*gamma, subtract b*gamma so that the
    # integrated noise spectrum agrees with the robust spatial estimate.
    baseline_offset = (noise_shape_variance - noise_variance) / particle_shape_variance
    calibrated_noise = np.maximum(
        noise_psd - baseline_offset * particle_psd,
        0.0,
    )

    # Clipping can slightly change the variance, so enforce the measured
    # noise scale once more.
    calibrated_noise_variance = radial_psd_to_variance(
        radial_points,
        calibrated_noise,
    )
    if noise_variance > eps:
        if calibrated_noise_variance <= eps:
            raise ValueError("calibrated noise spectrum is degenerate")
        calibrated_noise *= noise_variance / calibrated_noise_variance

    signal_variance = max(mean_patch_variance - noise_variance, 0.0)
    calibrated_particle = particle_psd * signal_variance / particle_shape_variance
    return calibrated_particle, calibrated_noise


def prewhiten_tomogram(
    tomogram: jax.Array | npt.ArrayLike,
    radial_points: jax.Array | npt.ArrayLike,
    noise_psd: jax.Array | npt.ArrayLike,
    regularization_fraction: float = 0.1,
) -> jax.Array:
    """Prewhiten a 3D tomogram from an isotropic radial noise PSD.

    Frequencies outside the spherical Nyquist band use the final radial PSD
    value, matching the boundary extension in the original picker.  A small
    median-spectrum regularizer prevents unstable amplification at spectral
    zeros.
    """
    if regularization_fraction < 0:
        raise ValueError("regularization_fraction must be non-negative")

    tomogram = jnp.asarray(
        tomogram,
        dtype=jnp.result_type(tomogram, jnp.float32),
    )
    radial_points = jnp.asarray(radial_points)
    noise_psd = jnp.asarray(noise_psd)
    if tomogram.ndim != 3:
        raise ValueError("tomogram must be three-dimensional")
    if radial_points.ndim != 1 or noise_psd.ndim != 1:
        raise ValueError("radial_points and noise_psd must be one-dimensional")
    if radial_points.shape != noise_psd.shape:
        raise ValueError("radial_points and noise_psd must have equal length")
    if radial_points.size < 2:
        raise ValueError("at least two radial PSD samples are required")

    stable_noise_psd = jnp.maximum(noise_psd, 0)
    stable_noise_psd = stable_noise_psd + (
        regularization_fraction * jnp.median(stable_noise_psd)
    )

    frequencies = [2 * jnp.pi * jnp.fft.fftfreq(size, d=1.0) for size in tomogram.shape]
    wx, wy, wz = jnp.meshgrid(*frequencies, indexing="ij")
    radius = jnp.sqrt(wx**2 + wy**2 + wz**2)
    radius = jnp.minimum(radius, radial_points[-1])
    noise_tensor = jnp.interp(radius, radial_points, stable_noise_psd)

    floor = jnp.finfo(tomogram.dtype).eps * jnp.maximum(
        jnp.max(noise_tensor),
        1.0,
    )
    inverse_sqrt_noise = jnp.where(
        noise_tensor > floor,
        jax.lax.rsqrt(noise_tensor),
        0,
    )

    centered = tomogram - jnp.mean(tomogram)
    whitened = jnp.fft.ifftn(jnp.fft.fftn(centered) * inverse_sqrt_noise).real
    whitened -= jnp.mean(whitened)
    whitened_norm = jnp.linalg.norm(whitened)
    return jnp.where(
        whitened_norm > 0,
        whitened / whitened_norm,
        whitened,
    )


def expand_spherical_harmonic_templates(
    radial_templates: npt.ArrayLike,
    orders: npt.ArrayLike,
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    z: npt.ArrayLike,
) -> tuple[
    npt.NDArray[np.generic],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
]:
    """Expand radial eigenfunctions into complete spherical-harmonic multiplets.

    For every radial eigenfunction of angular order ``ell``, this creates all
    ``2*ell + 1`` complex modes with ``m=-ell,...,ell``. SciPy's modern
    ``sph_harm_y`` interface expects polar colatitude followed by azimuth.
    """
    radial_templates = np.asarray(radial_templates)
    orders = np.asarray(orders, dtype=np.int64)
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    if radial_templates.ndim < 2:
        raise ValueError("radial_templates must have a leading mode dimension")
    if orders.ndim != 1 or orders.size != radial_templates.shape[0]:
        raise ValueError("orders must contain one entry per radial template")
    if np.any(orders < 0):
        raise ValueError("spherical-harmonic orders must be non-negative")
    if (
        x.shape != radial_templates.shape[1:]
        or y.shape != x.shape
        or z.shape != x.shape
    ):
        raise ValueError("coordinate grids must match the template grid")

    azimuth = np.arctan2(y, x)
    colatitude = np.arctan2(np.hypot(x, y), z)

    multiplicities = 2 * orders + 1
    radial_indices = np.repeat(
        np.arange(orders.size, dtype=np.int64),
        multiplicities,
    )
    mode_orders = orders[radial_indices]
    m_values = np.concatenate(
        [np.arange(-ell, ell + 1, dtype=np.int64) for ell in orders]
    )
    angle_axes = (slice(None),) + (None,) * colatitude.ndim
    modern_spherical_harmonic = getattr(
        special,
        "sph_harm_y",
        None,
    )
    if modern_spherical_harmonic is not None:
        angular_modes = modern_spherical_harmonic(
            mode_orders[angle_axes],
            m_values[angle_axes],
            colatitude[None, ...],
            azimuth[None, ...],
        )
    else:
        # SciPy < 1.15 used the reversed argument and angle order.
        legacy_spherical_harmonic = getattr(
            special,
            "sph_harm",
            None,
        )
        if legacy_spherical_harmonic is None:
            raise RuntimeError("SciPy provides neither sph_harm_y nor sph_harm")
        angular_modes = legacy_spherical_harmonic(
            m_values[angle_axes],
            mode_orders[angle_axes],
            azimuth[None, ...],
            colatitude[None, ...],
        )
    templates = radial_templates[radial_indices] * angular_modes

    return (
        templates,
        radial_indices,
        m_values,
    )


def radial_mode_truncation_index(
    eigvals: npt.ArrayLike,
    orders: npt.ArrayLike,
    energy_fraction: float = 0.99,
) -> int:
    """Return a complete-multiplet cutoff for a radial KLT spectrum."""
    eigvals = np.asarray(eigvals, dtype=np.float64)
    orders = np.asarray(orders, dtype=np.int64)
    if eigvals.ndim != 1 or orders.ndim != 1 or eigvals.shape != orders.shape:
        raise ValueError("eigvals and orders must be equal-length vectors")
    if eigvals.size == 0:
        raise ValueError("at least one eigenvalue is required")
    if np.any(eigvals < 0) or np.any(orders < 0):
        raise ValueError("eigvals and orders must be non-negative")
    if not 0 < energy_fraction <= 1:
        raise ValueError("energy_fraction must be in (0, 1]")

    mode_energy = eigvals * (2 * orders + 1)
    total_mode_energy = np.sum(mode_energy)
    if not np.isfinite(total_mode_energy) or total_mode_energy <= 0:
        raise ValueError("KLT eigenvalues have no positive finite energy")
    cumulative_energy = np.cumsum(mode_energy) / total_mode_energy
    return int(
        np.searchsorted(
            cumulative_energy,
            energy_fraction,
            side="left",
        )
        + 1
    )


def generate_uniform_radial_sampling_points(
    L: int,
    r_max: float,
    nbins: int | None = None,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.int32],
    npt.NDArray[np.int64],
]:
    """Create radial shells on a centered 3D angular-frequency lattice.

    Frequencies are measured in radians per voxel. Only samples inside the
    spherical bandlimit are assigned to a shell; cube-corner frequencies
    outside that ball receive shell id ``-1`` and are excluded.
    """
    if L < 3:
        raise ValueError("L must be at least 3")
    if r_max <= 0:
        raise ValueError("r_max must be positive")

    k = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(L, d=1.0))
    kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
    r = np.sqrt(kx**2 + ky**2 + kz**2)

    if nbins is None:
        # For an odd L=2*M-1 spectrum, this yields M radial samples from
        # zero through the Nyquist bandlimit, matching the original picker.
        nbins = (L + 1) // 2
    if nbins < 1:
        raise ValueError("nbins must be positive")

    if nbins == 1:
        uniform_points = np.array([0.0])
        r_edges = np.array([-r_max, r_max])
    else:
        uniform_points = np.linspace(0.0, r_max, nbins)
        dr = uniform_points[1] - uniform_points[0]
        r_edges = np.concatenate(
            (
                [uniform_points[0] - dr / 2],
                uniform_points[:-1] + dr / 2,
                [uniform_points[-1] + dr / 2],
            )
        )

    inside_bandlimit = r <= r_max + np.finfo(float).eps * max(1.0, r_max)
    shell_ids = np.full(r.shape, -1, dtype=np.int32)
    shell_ids[inside_bandlimit] = np.digitize(
        r[inside_bandlimit],
        r_edges[1:-1],
        right=False,
    ).astype(np.int32)

    counts = np.bincount(
        shell_ids[inside_bandlimit].ravel(),
        minlength=nbins,
    )
    return uniform_points, shell_ids, counts


def trigonometric_interpolation(
    x: jax.Array | npt.ArrayLike,
    y: jax.Array | npt.ArrayLike,
    z: jax.Array | npt.ArrayLike,
) -> jax.Array:
    """Interpolate periodic samples with the trigonometric cardinal kernel.

    Args:
        x: Uniform one-dimensional sample locations.
        y: Values at ``x`` with shape ``(num_samples,)``.
        z: One-dimensional query locations.

    Returns:
        Interpolated values at ``z``.

    Raises:
        ValueError: If the inputs are not compatible one-dimensional arrays.
    """
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    z = jnp.asarray(z)
    if x.ndim != 1 or y.ndim != 1 or z.ndim != 1:
        raise ValueError("x, y, and z must be one-dimensional")
    if x.shape != y.shape:
        raise ValueError("x and y must have equal length")
    if x.size < 2:
        raise ValueError("at least two interpolation samples are required")
    n = x.shape[0]

    scale = (x[1] - x[0]) * n / 2
    x_scaled = (x / scale) * jnp.pi / 2
    z_scaled = (z / scale) * jnp.pi / 2

    delta = z_scaled[:, None] - x_scaled[None, :]
    at_node = jnp.isclose(delta, 0)
    safe_delta = jnp.where(at_node, 1.0, delta)
    if n % 2:
        interpolation_matrix = jnp.sin(n * safe_delta) / (n * jnp.sin(safe_delta))
    else:
        interpolation_matrix = jnp.sin(n * safe_delta) / (n * jnp.tan(safe_delta))
    interpolation_matrix = jnp.where(
        at_node,
        1.0,
        interpolation_matrix,
    )

    interpolated = interpolation_matrix @ y

    # When a requested point is exactly an input node, return its sample
    # directly. This preserves the cardinal property under JAX's float32
    # default instead of accumulating O(eps) errors from sin(k*pi).
    exact_matches = z[:, None] == x[None, :]
    has_exact_match = jnp.any(exact_matches, axis=1)
    exact_indices = jnp.argmax(exact_matches, axis=1)
    return jnp.where(
        has_exact_match,
        y[exact_indices],
        interpolated,
    )


def generate_legendre_points(
    n: int,
    a: float,
    b: float,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Map Gauss-Legendre quadrature points to an interval.

    Args:
        n: Number of quadrature points.
        a: Interval start.
        b: Interval end.

    Returns:
        Descending quadrature points and their corresponding weights.

    Raises:
        ValueError: If ``n`` is not positive or the interval is invalid.
    """
    if n < 1:
        raise ValueError("n must be positive")
    if b <= a:
        raise ValueError("b must be greater than a")
    canonical_points, canonical_weights = special.roots_legendre(n)
    half_width = (b - a) / 2
    midpoint = (a + b) / 2
    points = half_width * canonical_points + midpoint
    weights = half_width * canonical_weights
    return np.flipud(points), weights
