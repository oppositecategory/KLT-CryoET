"""Nonnegative rank-one factorization of radial power spectra."""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

_DEFAULT_RANDOM_SEED = 1701


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "alpha_prev",
        "gamma_prev",
        "v_prev",
        "alpha",
        "gamma",
        "v",
        "iter_num",
    ],
    meta_fields=[],
)
@dataclass
class RpsdFactorization:
    """State and result of the alternating least-squares iteration.

    Attributes:
        alpha_prev: Particle occupancy weights from the preceding iteration,
            with shape ``(num_samples,)``.
        gamma_prev: Particle RPSD from the preceding iteration, with shape
            ``(num_frequencies,)``.
        v_prev: Noise RPSD from the preceding iteration, with shape
            ``(num_frequencies,)``.
        alpha: Current particle occupancy weights.
        gamma: Current nonnegative particle RPSD.
        v: Current nonnegative noise RPSD.
        iter_num: Number of completed ALS iterations.
    """

    alpha_prev: jax.Array
    gamma_prev: jax.Array
    v_prev: jax.Array
    alpha: jax.Array
    gamma: jax.Array
    v: jax.Array
    iter_num: jax.Array | int = 0


# Backward compatibility for notebooks and downstream imports.
RPSDFactorization = RpsdFactorization


def _safe_projection(
    basis: jax.Array,
    residuals: jax.Array,
) -> jax.Array:
    """Project residual columns onto a basis without dividing by zero."""
    denominator = jnp.sum(jnp.square(basis))
    safe_denominator = jnp.maximum(
        denominator,
        jnp.finfo(basis.dtype).tiny,
    )
    return jnp.dot(basis, residuals) / safe_denominator


def _relative_change(current: jax.Array, previous: jax.Array) -> jax.Array:
    """Return a numerically safe relative L2 change."""
    denominator = jnp.maximum(
        jnp.linalg.norm(current),
        jnp.finfo(current.dtype).tiny,
    )
    return jnp.linalg.norm(current - previous) / denominator


def alternating_least_squares_solver(
    samples: jax.Array,
    max_iter: int,
    eps: float,
    *,
    key: jax.Array | None = None,
) -> RpsdFactorization:
    """Factor sample RPSDs into particle weights, particle PSD, and noise PSD.

    The model is

    ``samples[j, :] = alpha[j] * gamma[:] + v[:]``,

    where ``alpha`` lies in ``[0, 1]`` and ``gamma`` and ``v`` are
    nonnegative. The implementation uses JAX control flow so the iteration can
    execute on an accelerator without Python-side synchronization.

    Args:
        samples: Radial spectra with shape
            ``(num_samples, num_frequencies)``.
        max_iter: Maximum number of ALS iterations.
        eps: Relative convergence tolerance.
        key: Optional JAX random key used only for degenerate zero-vector
            recovery. A deterministic local key is used when omitted.

    Returns:
        Final factorization state.

    Raises:
        ValueError: If the inputs have invalid shape or parameter values.
    """
    samples = jnp.asarray(
        samples,
        dtype=jnp.result_type(samples, jnp.float32),
    )
    if samples.ndim != 2:
        raise ValueError("samples must have shape (num_samples, num_frequencies)")
    if 0 in samples.shape:
        raise ValueError("samples must have nonempty sample and frequency axes")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    if eps <= 0:
        raise ValueError("eps must be positive")

    num_samples, num_frequencies = samples.shape
    spectra = samples.T
    key = jax.random.key(_DEFAULT_RANDOM_SEED) if key is None else key

    l1_norms = jnp.sum(jnp.abs(spectra), axis=0)
    noise_psd = jnp.abs(spectra[:, jnp.argmin(l1_norms)])

    # The difference between the spectra with extreme L-infinity norms is a
    # cheap, robust initialization for the nonnegative particle component.
    linf_norms = jnp.max(jnp.abs(spectra), axis=0)
    maximum_index = jnp.argmax(linf_norms)
    minimum_index = jnp.argmin(linf_norms)
    particle_psd = jnp.abs(spectra[:, maximum_index] - spectra[:, minimum_index])
    weights = jnp.clip(
        _safe_projection(
            particle_psd,
            spectra - noise_psd[:, None],
        ),
        0,
        1,
    )

    def has_not_converged(state: RpsdFactorization) -> jax.Array:
        errors = jnp.stack(
            (
                _relative_change(state.v, state.v_prev),
                _relative_change(state.gamma, state.gamma_prev),
                _relative_change(state.alpha, state.alpha_prev),
            )
        )
        return jnp.logical_and(
            jnp.any(errors >= eps),
            state.iter_num < max_iter,
        )

    def update(state: RpsdFactorization) -> RpsdFactorization:
        iteration_key = jax.random.fold_in(key, state.iter_num)
        weights_key, particle_key = jax.random.split(iteration_key)

        stable_weights = jax.lax.cond(
            jnp.linalg.norm(state.alpha) == 0,
            lambda _: jax.random.uniform(
                weights_key,
                (num_samples,),
                dtype=samples.dtype,
            ),
            lambda value: value,
            state.alpha,
        )

        new_particle_psd = jnp.maximum(
            _safe_projection(
                stable_weights,
                spectra.T - state.v[None, :],
            ),
            0,
        )
        new_noise_psd = jnp.maximum(
            jnp.mean(
                spectra - jnp.outer(new_particle_psd, stable_weights),
                axis=1,
            ),
            0,
        )
        stable_particle_psd = jax.lax.cond(
            jnp.linalg.norm(new_particle_psd) == 0,
            lambda _: jax.random.uniform(
                particle_key,
                (num_frequencies,),
                dtype=samples.dtype,
            ),
            lambda value: value,
            new_particle_psd,
        )
        new_weights = jnp.clip(
            _safe_projection(
                stable_particle_psd,
                spectra - new_noise_psd[:, None],
            ),
            0,
            1,
        )

        return RpsdFactorization(
            alpha_prev=stable_weights,
            gamma_prev=state.gamma,
            v_prev=state.v,
            alpha=new_weights,
            gamma=stable_particle_psd,
            v=new_noise_psd,
            iter_num=state.iter_num + 1,
        )

    initial_state = RpsdFactorization(
        alpha_prev=jnp.zeros_like(weights),
        gamma_prev=jnp.zeros_like(particle_psd),
        v_prev=jnp.zeros_like(noise_psd),
        alpha=weights,
        gamma=particle_psd,
        v=noise_psd,
    )
    return jax.lax.while_loop(has_not_converged, update, initial_state)
