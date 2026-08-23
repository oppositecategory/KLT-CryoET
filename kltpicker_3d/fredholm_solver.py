"""Nyström solver for the radial three-dimensional KLT equation."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.special import roots_legendre, spherical_jn


# NumPy's inverse Fourier transform represents the continuum convention
# C(h) = (2*pi)^-3 integral G(omega) exp(i omega.h) d omega.  The radial PSD
# calibration uses the same convention when converting spectral mass to
# spatial variance, so every covariance integral must carry this factor.
INVERSE_FOURIER_NORMALIZATION_3D = (2 * np.pi) ** -3


def _build_radial_fredholm_matrices(
    Gx: npt.ArrayLike,
    N: int,
    a: float,
    c: float,
    K: int,
) -> tuple[npt.NDArray[np.generic], npt.NDArray[np.float64]]:
    """Construct the radial covariance kernel and spatial quadrature matrix.

    The separated three-dimensional plane-wave basis for angular order ``N``
    is ``4 pi i**N j_N(r rho)``. Integrating that basis against the particle
    PSD produces a Hermitian radial covariance kernel.

    Args:
        Gx: Particle PSD sampled at the ``K`` frequency quadrature nodes.
        N: Nonnegative spherical-harmonic order.
        a: Spatial support radius, in voxels.
        c: Radial angular-frequency bandlimit, in radians per voxel.
        K: Number of Gauss-Legendre quadrature nodes.

    Returns:
        A pair containing the ``(K, K)`` covariance kernel and the diagonal
        ``(K, K)`` spatial quadrature matrix.

    Raises:
        ValueError: If the PSD or Fredholm parameters are invalid.
    """
    radial_psd = np.asarray(Gx)
    if K < 1:
        raise ValueError("K must be positive")
    if radial_psd.shape != (K,):
        raise ValueError(f"Gx must contain exactly K={K} radial PSD samples")
    if N < 0:
        raise ValueError("N must be nonnegative")
    if a <= 0 or c <= 0:
        raise ValueError("a and c must be positive")
    if not np.all(np.isfinite(radial_psd)):
        raise ValueError("Gx must contain only finite values")
    if np.any(radial_psd < 0):
        raise ValueError("Gx must be nonnegative")

    legendre_nodes, legendre_weights = roots_legendre(K)
    frequency_nodes = (c / 2) * (legendre_nodes + 1)
    spatial_nodes = (a / 2) * (legendre_nodes + 1)

    radial_frequency_grid = np.outer(spatial_nodes, frequency_nodes)
    basis = np.asarray(
        4
        * np.pi
        * (1j**N)
        * spherical_jn(N, radial_frequency_grid),
        dtype=np.complex128,
    )
    frequency_measure = (
        INVERSE_FOURIER_NORMALIZATION_3D
        * (c / 2)
        * legendre_weights
        * radial_psd
        * frequency_nodes**2
    )
    kernel = (basis * frequency_measure[None, :]) @ np.conj(basis).T
    kernel = (kernel + kernel.conj().T) / 2

    spatial_measure = (a / 2) * legendre_weights * spatial_nodes**2
    quadrature = np.diag(spatial_measure)
    return np.real_if_close(kernel), quadrature


def solve_radial_fredholm_equation(
    Gx: npt.ArrayLike,
    N: int,
    a: float,
    c: float,
    K: int = 150,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.generic],
    npt.NDArray[np.float64],
]:
    """Solve the radial KLT integral equation with the Nyström method.

    Parameter names follow the notation used by the original implementation
    and are retained for backward compatibility.

    Args:
        Gx: Particle PSD at the frequency quadrature nodes.
        N: Spherical-harmonic order.
        a: Spatial template-support radius, in voxels.
        c: Angular-frequency bandlimit, in radians per voxel.
        K: Gauss-Legendre quadrature order.

    Returns:
        A tuple containing descending eigenvalues, radial eigenfunctions
        stored in columns, and the spatial quadrature matrix.
    """
    kernel, quadrature = _build_radial_fredholm_matrices(
        Gx,
        N,
        a,
        c,
        K,
    )

    # The direct Nyström matrix is kernel @ quadrature. Its similar Hermitian
    # form preserves the same eigenvalues and weighted orthogonality.
    spatial_weights = np.diag(quadrature)
    sqrt_spatial_weights = np.sqrt(spatial_weights)
    hermitian_operator = (
        sqrt_spatial_weights[:, None] * kernel * sqrt_spatial_weights[None, :]
    )
    hermitian_operator = (hermitian_operator + hermitian_operator.conj().T) / 2
    eigenvalues, weighted_eigenvectors = np.linalg.eigh(hermitian_operator)
    eigenfunctions = weighted_eigenvectors / sqrt_spatial_weights[:, None]

    eigenvalues = eigenvalues[::-1]
    eigenfunctions = np.real_if_close(eigenfunctions[:, ::-1])

    tolerance = np.finfo(float).eps * K * max(1.0, np.max(np.abs(eigenvalues)))
    eigenvalues = np.where(
        np.abs(eigenvalues) < tolerance,
        0,
        eigenvalues,
    )
    eigenfunctions[:, eigenvalues == 0] = 0
    return eigenvalues, eigenfunctions, quadrature
