import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose

from kltpicker_3d.tomogram import KLTParticleDetector3D
from kltpicker_3d.utils import expand_spherical_harmonic_templates


def test_nonnegative_spherical_harmonics_match_full_expansion():
    grid = np.arange(-1, 2)
    z, y, x = np.meshgrid(grid, grid, grid, indexing="ij")
    radial_templates = np.stack(
        (
            np.ones_like(x, dtype=np.float32),
            np.full_like(x, 2, dtype=np.float32),
        )
    )
    orders = np.asarray([1, 2])

    full, full_radial_indices, full_m_values = (
        expand_spherical_harmonic_templates(
            radial_templates,
            orders,
            x,
            y,
            z,
        )
    )
    nonnegative, radial_indices, m_values = (
        expand_spherical_harmonic_templates(
            radial_templates,
            orders,
            x,
            y,
            z,
            nonnegative_m_only=True,
        )
    )

    keep = full_m_values >= 0
    assert_allclose(nonnegative, full[keep])
    assert np.array_equal(radial_indices, full_radial_indices[keep])
    assert np.array_equal(m_values, full_m_values[keep])


def test_complete_detector_executes_with_cpu_jax():
    rng = np.random.default_rng(1701)
    shape = (25, 25, 25)
    grid = np.indices(shape, dtype=np.float32)
    tomogram = 0.30 * rng.standard_normal(shape).astype(np.float32)
    truth = np.array([[8, 8, 8], [17, 17, 16]])

    for center in truth:
        radius_squared = sum(
            (grid[axis] - center[axis]) ** 2
            for axis in range(3)
        )
        tomogram += np.exp(
            -radius_squared / (2 * 1.25**2)
        ).astype(np.float32)

    detector = KLTParticleDetector3D(
        jnp.asarray(tomogram),
        particle_diameter=7.0,
        mgscale=1.0,
        num_particles=2,
        legendre_order=8,
        threshold=-1.0,
        max_iter=30,
        max_order=3,
    )
    num_detected, picked = detector.process_tomogram()
    jax.block_until_ready(detector.whitened_tomogram)

    assert num_detected == 2
    assert detector.score_mat.shape == (19, 19, 19)
    assert np.isfinite(detector.score_mat).all()
    assert np.isfinite(np.asarray(detector.whitened_tomogram)).all()
    assert detector.noise_variance > 0
    assert detector.patch_size == 5
    assert np.isclose(detector.fredholm_radius_voxels, 2.8)
    assert detector.template_side == 7

    template_grid = np.arange(-3, 4)
    z, y, x = np.meshgrid(
        template_grid,
        template_grid,
        template_grid,
        indexing="ij",
    )
    outside_support = np.sqrt(z**2 + y**2 + x**2) > 2.8
    assert np.all(np.asarray(detector.templates)[:, outside_support] == 0)

    # The deterministic smoke volume should localize both centers to within
    # one voxel in each coordinate.
    picked_centers = picked[:, :3]
    picked_centers = picked_centers[np.argsort(picked_centers[:, 0])]
    truth = truth[np.argsort(truth[:, 0])]
    assert_allclose(picked_centers, truth, atol=1.0)

    # ell=0,1,2 multiplets are complete for the retained radial modes.
    orders, counts = np.unique(
        detector.template_orders,
        return_counts=True,
    )
    assert np.all(counts % (2 * orders + 1) == 0)
