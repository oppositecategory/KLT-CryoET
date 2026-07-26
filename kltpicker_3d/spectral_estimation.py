"""Isotropic autocorrelation and power-spectrum estimation in three dimensions."""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from skimage.filters import window


def estimate_isotropic_powerspectrum_tensor(
    tomogram: jax.Array | npt.ArrayLike,
    max_d: int,
) -> jax.Array:
    """Estimate an isotropic 3D power-spectrum tensor from a cubic patch.

    The estimator radially averages the unbiased spatial autocorrelation,
    applies a Gaussian lag window, and Fourier transforms the resulting
    centrosymmetric tensor. The spectrum is normalized to preserve the
    patch's mean-centered spatial energy.

    Args:
        tomogram: Cubic real-valued patch with shape ``(size, size, size)``.
        max_d: Maximum ACF lag, in voxels. Values at least as large as the
            patch are clipped to ``size - 1``.

    Returns:
        Nonnegative centered power-spectrum tensor with shape
        ``(2 * size - 1,) * 3``.

    Raises:
        ValueError: If the patch is not a nonempty cube or ``max_d`` is not
            positive.
    """
    patch = jnp.asarray(tomogram)
    if patch.ndim != 3 or len(set(patch.shape)) != 1:
        raise ValueError("tomogram must be a nonempty cubic 3D array")
    size = patch.shape[0]
    if size < 1:
        raise ValueError("tomogram must be nonempty")
    if max_d < 1:
        raise ValueError("max_d must be positive")
    max_distance = min(max_d, size - 1)

    autocorrelation = estimate_isotropic_autocorrelation(
        patch,
        max_distance,
    )
    spectrum_shape = (2 * size - 1,) * 3
    lag_window = jnp.asarray(
        window(("gaussian", max_distance), spectrum_shape),
        dtype=patch.real.dtype,
    )

    # A real centrosymmetric ACF has a real Fourier transform up to numerical
    # roundoff. Clipping removes small negative values caused by truncation.
    power_spectrum = jnp.maximum(
        jnp.real(cfftn(autocorrelation * lag_window)),
        0,
    )
    mean_energy = jnp.mean(jnp.square(patch - jnp.mean(patch)))
    spectral_mass = jnp.sum(power_spectrum)
    safe_mass = jnp.maximum(
        spectral_mass,
        jnp.finfo(power_spectrum.dtype).tiny,
    )
    normalized_spectrum = power_spectrum * mean_energy * power_spectrum.size / safe_mass
    return jnp.where(spectral_mass > 0, normalized_spectrum, 0)


def estimate_isotropic_autocorrelation(
    tomogram: jax.Array | npt.ArrayLike,
    max_d: int,
) -> jax.Array:
    """Estimate an unbiased isotropic ACF on a full centered lag grid.

    Args:
        tomogram: Cubic patch with shape ``(size, size, size)``.
        max_d: Maximum retained radial lag, in voxels.

    Returns:
        Real centrosymmetric ACF with shape ``(2 * size - 1,) * 3``.

    Raises:
        ValueError: If the input shape or maximum distance is invalid.
    """
    patch = jnp.asarray(tomogram)
    if patch.ndim != 3 or len(set(patch.shape)) != 1:
        raise ValueError("tomogram must be a nonempty cubic 3D array")
    size = patch.shape[0]
    if not 0 <= max_d < size:
        raise ValueError("max_d must lie in [0, patch_size)")

    # Lag geometry depends only on the static patch shape and max_d. Building
    # it with NumPy keeps the resulting index-array sizes static under jit,
    # vmap, and scan; data-dependent jnp.where/jnp.unique sizes cannot be
    # staged by JAX.
    positive_lags = np.arange(max_d + 1)
    lag_z_numpy, lag_y_numpy, lag_x_numpy = np.meshgrid(
        positive_lags,
        positive_lags,
        positive_lags,
        indexing="ij",
    )
    squared_radius_numpy = (
        lag_z_numpy**2 + lag_y_numpy**2 + lag_x_numpy**2
    )
    valid_lags_numpy = np.where(squared_radius_numpy <= max_d**2)
    valid_lags = tuple(
        jnp.asarray(indices)
        for indices in valid_lags_numpy
    )
    squared_distances = jnp.asarray(
        np.unique(squared_radius_numpy[valid_lags_numpy])
    )
    lag_z = jnp.asarray(lag_z_numpy)
    lag_y = jnp.asarray(lag_y_numpy)
    lag_x = jnp.asarray(lag_x_numpy)
    squared_radius = jnp.asarray(squared_radius_numpy)

    radial_indices = jnp.searchsorted(
        squared_distances,
        squared_radius,
        side="left",
    )
    distance_map = jnp.zeros(
        squared_radius.shape,
        dtype=jnp.int32,
    )
    distance_map = distance_map.at[valid_lags].set(radial_indices[valid_lags])

    # The number of voxel pairs separated by a positive lag is analytic,
    # avoiding a second FFT of an all-ones volume.
    overlap_counts = ((size - lag_z) * (size - lag_y) * (size - lag_x)).astype(
        patch.real.dtype
    )

    padded_shape = (2 * size - 1,) * 3
    padded_patch = jnp.zeros(
        padded_shape,
        dtype=jnp.result_type(patch, jnp.complex64),
    )
    padded_patch = padded_patch.at[:size, :size, :size].set(patch)
    patch_autocorrelation = calculate_autocorrelation(padded_patch).real

    initial_accumulator = jnp.zeros(
        (2, squared_distances.shape[0]),
        dtype=patch_autocorrelation.dtype,
    )
    radial_sum, radial_count = accumulate_acf_radially(
        patch_autocorrelation,
        distance_map,
        valid_lags,
        overlap_counts,
        initial_accumulator,
    )
    radial_autocorrelation = jnp.where(
        radial_count > 0,
        radial_sum / radial_count,
        0,
    )
    return create_autocorrelation_tensor(
        radial_autocorrelation,
        squared_distances,
        size,
        max_d,
    )


def create_autocorrelation_tensor(
    r: jax.Array | npt.ArrayLike,
    dists: jax.Array | npt.ArrayLike,
    N: int,
    max_d: int,
) -> jax.Array:
    """Embed radial ACF samples on a centered, symmetric 3D grid.

    Parameter names are retained for compatibility with existing notebooks.

    Args:
        r: Radial autocorrelation samples.
        dists: Squared integer radii corresponding to ``r``.
        N: Original cubic patch size.
        max_d: Maximum supported spatial lag, in voxels.

    Returns:
        Centrosymmetric tensor with shape ``(2 * N - 1,) * 3``.
    """
    radial_values = jnp.asarray(r)
    squared_distances = jnp.asarray(dists)
    grid = jnp.arange(-(N - 1), N)
    lag_z, lag_y, lag_x = jnp.meshgrid(
        grid,
        grid,
        grid,
        indexing="ij",
    )
    squared_radius = lag_z**2 + lag_y**2 + lag_x**2
    radial_index = jnp.searchsorted(
        squared_distances,
        squared_radius,
        side="left",
    )
    safe_index = jnp.minimum(
        radial_index,
        squared_distances.size - 1,
    )
    inside_support = (
        (squared_radius <= max_d**2)
        & (radial_index < squared_distances.size)
        & (squared_distances[safe_index] == squared_radius)
    )
    autocorrelation = jnp.where(
        inside_support,
        radial_values[safe_index],
        0,
    )

    # Keep the projection explicit so future changes cannot introduce an odd
    # ACF component and therefore a spurious imaginary Fourier component.
    return 0.5 * (autocorrelation + jnp.flip(autocorrelation, axis=(0, 1, 2)))


def cfftn(values: jax.Array | npt.ArrayLike) -> jax.Array:
    """Return the centered N-dimensional discrete Fourier transform."""
    values = jnp.asarray(values)
    return jnp.fft.fftshift(jnp.fft.fftn(jnp.fft.ifftshift(values)))


def calculate_autocorrelation(
    patch: jax.Array | npt.ArrayLike,
) -> jax.Array:
    """Calculate a cyclic autocorrelation using Wiener-Khinchin."""
    patch_fft = jnp.fft.fftn(jnp.asarray(patch))
    return jnp.fft.ifftn(patch_fft * jnp.conj(patch_fft))


def accumulate_acf_radially(
    acf: jax.Array,
    dist_map: jax.Array,
    valid_dists: Sequence[jax.Array],
    dists_counts: jax.Array,
    init: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Accumulate ACF values and overlap counts by squared radial distance.

    Args:
        acf: Autocorrelation tensor.
        dist_map: Map from positive-lag coordinates to radial-bin indices.
        valid_dists: Tuple of coordinate arrays identifying supported lags.
        dists_counts: Number of overlapping voxel pairs at every positive lag.
        init: Initial ``(2, num_radial_bins)`` accumulator.

    Returns:
        Radial ACF sums and corresponding overlap-count sums.
    """
    radial_bins = dist_map[tuple(valid_dists)]
    radial_sum = init[0].at[radial_bins].add(acf[tuple(valid_dists)])
    radial_count = init[1].at[radial_bins].add(dists_counts[tuple(valid_dists)])
    return radial_sum, radial_count
