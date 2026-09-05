"""Benchmark scoring FFT geometry and template batch memory on one GPU.

Run every configuration in a separate process with JAX preallocation disabled.
The workload always scores eight compact templates so timings across template
batch sizes include the same arithmetic. Compilation and cuFFT-plan warm-up are
excluded from the reported steady-state timings.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


LOADED_SIDE = 358
COMPACT_TEMPLATE_SIDE = 97
WHITENING_SIDE = 75
CORE_SIDE = 186
SCORE_HALO = 1
OUTPUT_SIDE = CORE_SIDE + 2 * SCORE_HALO
CROP_START = (WHITENING_SIDE - 1) // 2 + (COMPACT_TEMPLATE_SIDE - 1) // 2
TEMPLATE_COUNT = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fft-side", type=int, choices=(358, 360), required=True)
    parser.add_argument("--batch-size", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output-score", type=Path)
    return parser.parse_args()


def centered_filter_spectrum(filters: jax.Array, fft_side: int) -> jax.Array:
    padding = (
        (0, 0),
        (0, fft_side - filters.shape[-3]),
        (0, fft_side - filters.shape[-2]),
        (0, fft_side - filters.shape[-1]),
    )
    padded = jnp.pad(filters, padding)
    radii = tuple((size - 1) // 2 for size in filters.shape[-3:])
    return jnp.fft.fftn(
        jnp.roll(padded, shift=tuple(-r for r in radii), axis=(-3, -2, -1)),
        axes=(-3, -2, -1),
    )


def score_eight_templates(
    loaded_subvolume: jax.Array,
    whitening_filter: jax.Array,
    templates: jax.Array,
    weights: jax.Array,
    *,
    fft_side: int,
    batch_size: int,
) -> jax.Array:
    extra = fft_side - LOADED_SIDE
    padded_volume = jnp.pad(
        loaded_subvolume,
        ((0, extra), (0, extra), (0, extra)),
    )
    volume_spectrum = jnp.fft.fftn(padded_volume)
    whitening_spectrum = centered_filter_spectrum(
        whitening_filter[None, ...], fft_side
    )[0]
    whitened_spectrum = volume_spectrum * whitening_spectrum
    initial_score = jnp.zeros((OUTPUT_SIDE,) * 3, dtype=jnp.float32)

    def accumulate(batch_index: jax.Array, score: jax.Array) -> jax.Array:
        start = batch_index * batch_size
        template_batch = jax.lax.dynamic_slice_in_dim(
            templates, start, batch_size, axis=0
        )
        weight_batch = jax.lax.dynamic_slice_in_dim(
            weights, start, batch_size, axis=0
        )
        template_spectra = centered_filter_spectrum(template_batch, fft_side)
        responses = jnp.fft.ifftn(
            whitened_spectrum[None, ...] * template_spectra,
            axes=(-3, -2, -1),
        )
        owned = jax.lax.dynamic_slice(
            responses,
            (0, CROP_START, CROP_START, CROP_START),
            (batch_size, OUTPUT_SIDE, OUTPUT_SIDE, OUTPUT_SIDE),
        )
        return score + jnp.sum(
            weight_batch[:, None, None, None]
            * (owned.real * owned.real + owned.imag * owned.imag),
            axis=0,
        )

    return jax.lax.fori_loop(
        0,
        TEMPLATE_COUNT // batch_size,
        accumulate,
        initial_score,
    )


def memory_analysis_dict(compiled: Any) -> dict[str, int | None]:
    analysis = compiled.memory_analysis()
    names = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "temp_size_in_bytes",
        "alias_size_in_bytes",
        "host_argument_size_in_bytes",
        "host_output_size_in_bytes",
        "host_temp_size_in_bytes",
    )
    return {name: getattr(analysis, name, None) for name in names}


def main() -> None:
    args = parse_args()
    if args.warmups < 1 or args.iterations < 1:
        raise ValueError("warmups and iterations must be positive")
    devices = jax.devices("gpu")
    if len(devices) != 1:
        raise RuntimeError(
            "Expose exactly one GPU with CUDA_VISIBLE_DEVICES before benchmarking"
        )
    device = devices[0]
    rng = np.random.default_rng(17)
    volume = jax.device_put(
        rng.standard_normal((LOADED_SIDE,) * 3, dtype=np.float32), device
    )
    whitening = np.zeros((WHITENING_SIDE,) * 3, dtype=np.float32)
    whitening[(WHITENING_SIDE // 2,) * 3] = 1.0
    whitening = jax.device_put(whitening, device)
    templates = jax.device_put(
        (
            rng.standard_normal(
                (TEMPLATE_COUNT, COMPACT_TEMPLATE_SIDE, COMPACT_TEMPLATE_SIDE,
                 COMPACT_TEMPLATE_SIDE),
                dtype=np.float32,
            )
            + 1j
            * rng.standard_normal(
                (TEMPLATE_COUNT, COMPACT_TEMPLATE_SIDE, COMPACT_TEMPLATE_SIDE,
                 COMPACT_TEMPLATE_SIDE),
                dtype=np.float32,
            )
        ).astype(np.complex64),
        device,
    )
    weights = jax.device_put(np.linspace(0.75, 1.25, TEMPLATE_COUNT, dtype=np.float32), device)

    function = jax.jit(
        lambda v, w, t, a: score_eight_templates(
            v,
            w,
            t,
            a,
            fft_side=args.fft_side,
            batch_size=args.batch_size,
        ),
        device=device,
    )
    compiled = function.lower(volume, whitening, templates, weights).compile()
    for _ in range(args.warmups):
        compiled(volume, whitening, templates, weights).block_until_ready()

    timings = []
    result = None
    for _ in range(args.iterations):
        started = time.perf_counter()
        result = compiled(volume, whitening, templates, weights)
        result.block_until_ready()
        timings.append(time.perf_counter() - started)
    assert result is not None
    host_result = np.asarray(result)
    if args.output_score is not None:
        args.output_score.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output_score, host_result)

    statistics = device.memory_stats() or {}
    timings_array = np.asarray(timings)
    summary = {
        "jax_version": jax.__version__,
        "device": device.device_kind,
        "fft_side": args.fft_side,
        "batch_size": args.batch_size,
        "template_count": TEMPLATE_COUNT,
        "median_seconds": float(np.median(timings_array)),
        "mean_seconds": float(np.mean(timings_array)),
        "p10_seconds": float(np.quantile(timings_array, 0.1)),
        "p90_seconds": float(np.quantile(timings_array, 0.9)),
        "median_milliseconds_per_template": float(
            1_000 * np.median(timings_array) / TEMPLATE_COUNT
        ),
        "templates_per_second": float(
            TEMPLATE_COUNT / np.median(timings_array)
        ),
        "compiled_memory": memory_analysis_dict(compiled),
        "runtime_bytes_in_use": statistics.get("bytes_in_use"),
        "runtime_peak_bytes_in_use": statistics.get("peak_bytes_in_use"),
        "score_min": float(np.min(host_result)),
        "score_max": float(np.max(host_result)),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
