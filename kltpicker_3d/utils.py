import jax 
import jax.numpy as jnp

import scipy
import numpy as np

from functools import partial


def ranked_local_maxima_nms_3d(score_volume, radius, max_picks):
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
    max_picks = int(max_picks)
    if max_picks < 0:
        raise ValueError("max_picks must be nonnegative")
    if max_picks == 0:
        return np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=scores.dtype)

    neighborhood_max = scipy.ndimage.maximum_filter(
        scores,
        size=3,
        mode="constant",
        cval=-np.inf,
    )
    neighborhood_min = scipy.ndimage.minimum_filter(
        scores,
        size=3,
        mode="constant",
        cval=np.inf,
    )
    candidate_mask = (
        np.isfinite(scores)
        & (scores == neighborhood_max)
        & (scores > neighborhood_min)
    )
    candidate_indices = np.argwhere(candidate_mask)
    if candidate_indices.size == 0:
        return np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=scores.dtype)

    candidate_values = scores[tuple(candidate_indices.T)]
    ranked = np.argsort(-candidate_values, kind="stable")
    candidate_indices = candidate_indices[ranked]
    candidate_values = candidate_values[ranked]

    accepted_indices = []
    accepted_values = []
    radius_squared = float(radius) ** 2
    for index, value in zip(candidate_indices, candidate_values):
        if accepted_indices:
            separation_squared = np.sum(
                (np.asarray(accepted_indices) - index) ** 2,
                axis=1,
            )
            if np.any(separation_squared <= radius_squared):
                continue
        accepted_indices.append(index)
        accepted_values.append(value)
        if len(accepted_indices) == max_picks:
            break

    return (
        np.asarray(accepted_indices, dtype=np.int64).reshape(-1, 3),
        np.asarray(accepted_values, dtype=scores.dtype),
    )


def bandpass_filter_3d(volume,
                       low_fraction=0.05,
                       high_fraction=0.05,
                       normalize=True):
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

    frequencies = [
        2 * jnp.pi * jnp.fft.fftfreq(size, d=1.0)
        for size in volume.shape
    ]
    wz, wy, wx = jnp.meshgrid(*frequencies, indexing="ij")
    radial_frequency = jnp.sqrt(wz**2 + wy**2 + wx**2)
    low_cutoff = low_fraction * jnp.pi
    high_cutoff = (1 - high_fraction) * jnp.pi
    passband = (
        (radial_frequency >= low_cutoff)
        & (radial_frequency <= high_cutoff)
    )

    filtered = jnp.fft.ifftn(
        jnp.fft.fftn(centered) * passband
    ).real
    return filtered - jnp.mean(filtered)


def radial_average_jax(X, shell_ids, counts, nbins):
    x = X.ravel()
    ids = shell_ids.ravel()
    mask = ids >= 0
    sums = jnp.bincount(ids[mask], weights=x[mask], length=nbins)
    return jnp.where(counts > 0, sums / counts, 0)

def prewhiten_patch(patch, noise_psd):
    """ Pre-whitening a patch using approximation of the noise RPSD.

        Args:
            patch: sub-tomogram of size LxLxL 
            noise_psd: tensor of size MxMxM containing the noise RPSD

        returns:
            p3: sub-tomogram after cleaning the approximated noise from it's spectrum
    """
    L,_,_ = patch.shape
    M,_,_ = noise_psd.shape
    midpoint = M//2

    start = midpoint - L//2
    end = midpoint + L//2 

    filter = jnp.sqrt(noise_psd)
    filter /= jnp.linalg.norm(filter)
    
    #Symmetrize the PSD across each axis
    filter = (filter + jnp.flip(filter,axis=0))/2
    filter = (filter + jnp.flip(filter,axis=1))/2
    filter = (filter + jnp.flip(filter,axis=2))/2

    mask = filter > 1e-14
    filter = jnp.where(mask, 1 / filter,0)

    padded = jnp.zeros_like(noise_psd)
    padded = padded.at[start:end, start:end, start:end].set(patch)

    fp = jnp.fft.fftshift(jnp.fft.fftn(jnp.fft.ifftshift(padded)))
    fp *= filter 
    pp2 = jnp.fft.fftshift(jnp.fft.ifftn(jnp.fft.ifftshift(fp)))
    p2 = pp2[start:end,start:end,start:end].real 
    return p2 

def radial_psd_to_variance(radial_points, radial_psd):
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
    spherical_integral = 4 * np.pi * np.sum(
        0.5 * (integrand[:-1] + integrand[1:]) * dr
    )
    sampled_ball_volume = 4 * np.pi * radial_points[-1]**3 / 3
    cube_volume = (2 * np.pi) ** 3
    corner_integral = (
        cube_volume - sampled_ball_volume
    ) * radial_psd[-1]
    return float(
        (spherical_integral + corner_integral) / cube_volume
    )

def calibrate_radial_psds(radial_points,
                          particle_psd,
                          noise_psd,
                          noise_variance,
                          mean_patch_variance):
    """Resolve the ALS scale/offset ambiguity using measured variances.

    ALS identifies the clean spectrum only up to a positive scale and can
    absorb a multiple of that spectrum into its noise baseline.  This is the
    3D counterpart of the calibration step in the original 2D picker.
    """
    particle_psd = np.asarray(particle_psd, dtype=np.float64)
    noise_psd = np.asarray(noise_psd, dtype=np.float64)
    noise_variance = float(noise_variance)
    mean_patch_variance = float(mean_patch_variance)

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
    baseline_offset = (
        noise_shape_variance - noise_variance
    ) / particle_shape_variance
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
    calibrated_particle = (
        particle_psd * signal_variance / particle_shape_variance
    )
    return calibrated_particle, calibrated_noise

def prewhiten_tomogram(tomogram,
                       radial_points,
                       noise_psd,
                       regularization_fraction=0.1):
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

    stable_noise_psd = jnp.maximum(noise_psd, 0)
    stable_noise_psd = stable_noise_psd + (
        regularization_fraction * jnp.median(stable_noise_psd)
    )

    frequencies = [
        2 * jnp.pi * jnp.fft.fftfreq(size, d=1.0)
        for size in tomogram.shape
    ]
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
    whitened = jnp.fft.ifftn(
        jnp.fft.fftn(centered) * inverse_sqrt_noise
    ).real
    whitened -= jnp.mean(whitened)
    whitened_norm = jnp.linalg.norm(whitened)
    return jnp.where(
        whitened_norm > 0,
        whitened / whitened_norm,
        whitened,
    )

def expand_spherical_harmonic_templates(radial_templates,
                                        orders,
                                        x,
                                        y,
                                        z):
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
    if x.shape != radial_templates.shape[1:] or y.shape != x.shape or z.shape != x.shape:
        raise ValueError("coordinate grids must match the template grid")

    azimuth = np.arctan2(y, x)
    colatitude = np.arctan2(np.hypot(x, y), z)

    templates = []
    radial_indices = []
    m_values = []
    for radial_index, ell in enumerate(orders):
        for m in range(-int(ell), int(ell) + 1):
            if hasattr(scipy.special, "sph_harm_y"):
                angular_mode = scipy.special.sph_harm_y(
                    int(ell),
                    m,
                    colatitude,
                    azimuth,
                )
            else:
                # SciPy < 1.15 used the reversed argument order and named the
                # angles azimuth/elevation, although the latter is colatitude.
                angular_mode = scipy.special.sph_harm(
                    m,
                    int(ell),
                    azimuth,
                    colatitude,
                )
            templates.append(radial_templates[radial_index] * angular_mode)
            radial_indices.append(radial_index)
            m_values.append(m)

    return (
        np.stack(templates, axis=0),
        np.asarray(radial_indices, dtype=np.int64),
        np.asarray(m_values, dtype=np.int64),
    )

def radial_mode_truncation_index(eigvals, orders, energy_fraction=0.99):
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

def generate_uniform_radial_sampling_points(L, r_max, nbins=None):
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
    nbins = int(nbins)
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

def trigonometric_interpolation(x,y,z):
    n = x.shape[0]
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    z = jnp.asarray(z)
    
    scale = (x[1] - x[0]) * n / 2 
    x_scaled = (x / scale) * jnp.pi / 2 
    z_scaled = (z / scale) * jnp.pi / 2

    delta = z_scaled[:, None] - x_scaled[None, :]
    at_node = jnp.isclose(delta, 0)
    safe_delta = jnp.where(at_node, 1.0, delta)
    if n % 2:
        M = jnp.sin(n * safe_delta) / (n * jnp.sin(safe_delta))
    else:
        M = jnp.sin(n * safe_delta) / (n * jnp.tan(safe_delta))
    M = jnp.where(at_node, 1.0, M)

    interpolated = M @ y

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

def generate_legendre_points(n, a, b):
    """
    Get n leggauss points in interval [a, b]
    Parameters
    ----------
    n : int
        Number of points.
    a : float
        Interval starting point.
    b : float
        Interval end point.
    Returns
    -------
    x : numpy.ndarray
        Sample points.
    w : numpy.ndarray
        Weights.
    """
    x1, w = scipy.special.roots_legendre(n)
    m = (b - a) / 2
    c = (a + b) / 2
    x = m * x1 + c
    w = m * w
    x = np.flipud(x)
    return x, w
