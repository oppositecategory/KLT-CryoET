import numpy as np
import scipy
from scipy.special import spherical_jn


def _build_radial_fredholm_matrices(Gx, N, a, c, K):
    """Construct the radial covariance kernel and spatial quadrature matrix."""
    Gx = np.asarray(Gx)
    if Gx.shape != (K,):
        raise ValueError(f"Gx must contain exactly K={K} radial PSD samples")
    if a <= 0 or c <= 0:
        raise ValueError("a and c must be positive")

    legendre_nodes, legendre_weights = scipy.special.roots_legendre(K)
    frequency_nodes = (c / 2) * (legendre_nodes + 1)
    spatial_nodes = (a / 2) * (legendre_nodes + 1)

    # In 3D, separating the plane wave expansion into spherical harmonics
    # produces 4*pi*i^N*j_N(r*rho). The covariance pairs this basis with its
    # complex conjugate, so the resulting kernel is Hermitian positive
    # semidefinite whenever Gx is non-negative.
    radial_frequency_grid = np.outer(spatial_nodes, frequency_nodes)
    basis = 4 * np.pi * (1j**N) * spherical_jn(N, radial_frequency_grid)

    frequency_measure = (
        (c / 2)
        * legendre_weights
        * Gx
        * frequency_nodes**2
    )
    kernel = (basis * frequency_measure[None, :]) @ basis.conj().T
    kernel = (kernel + kernel.conj().T) / 2

    spatial_measure = (
        (a / 2)
        * legendre_weights
        * spatial_nodes**2
    )
    quadrature = np.diag(spatial_measure)
    return np.real_if_close(kernel), quadrature


def solve_radial_fredholm_equation(Gx,
                           N: int,
                           a: float,
                           c: float,
                           K=150):
    """
        Solves radial KLT integral equation using Nystrom method.
        
        args:
            Gx: particle function's radial PSD
            a: radius of the spatial template support, in voxels
            c: particle function's bandlimit 
            K: Legendre quadrature order (bounds the n indices in the solutions)

        returns:
            eigvals: eigenvalues lambda_{N,n} for n below K
            eigfuncs: eigenfunctions R_{N,n} for n below K
    """
    H, W = _build_radial_fredholm_matrices(Gx, N, a, c, K)

    # H @ W is the Nyström discretization of the integral operator.
    # This similar Hermitian matrix has the same eigenvalues while preserving
    # the r^2-weighted orthogonality of the recovered radial eigenfunctions.
    sqrt_W = np.diag(np.sqrt(np.diag(W)))
    A = sqrt_W @ H @ sqrt_W
    A = (A + A.conj().T) / 2
    eigvals, Y = np.linalg.eigh(A)
    R = np.linalg.solve(sqrt_W, Y)

    eigvals = eigvals[::-1]
    eigfuncs = np.real_if_close(R[:,::-1])

    tolerance = np.finfo(float).eps * K * max(1.0, np.max(np.abs(eigvals)))
    eigvals = np.where(np.abs(eigvals) < tolerance, 0, eigvals)
    eigfuncs[:, eigvals == 0] = 0
    return eigvals, eigfuncs, W
