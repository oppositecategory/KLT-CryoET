"""Benchmark template FFT reuse across a resident subvolume tile.

Every invocation uses one visible GPU and one static tile size. Run schedules
in separate processes with JAX preallocation disabled so allocator peak-memory
statistics remain comparable.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


LOADED_SIDE = 358
FFT_SIDE = 360
COMPACT_TEMPLATE_SIDE = 97
WHITENING_SIDE = 75
CORE_SIDE = 186
OUTPUT_SIDE = CORE_SIDE + 2
CROP_START = (WHITENING_SIDE - 1) // 2 + (COMPACT_TEMPLATE_SIDE - 1) // 2
DEFAULT_TEMPLATE_COUNT = 8


def parse_args() -> argparse.Namespace:
    """Parse one isolated benchmark configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schedule",
        choices=("subvolume-major", "tiled-template-reuse"),
        required=True,
    )
    parser.add_argument("--tile-size", type=int, required=True)
    parser.add_argument("--template-count", type=int, default=DEFAULT_TEMPLATE_COUNT)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    return parser.parse_args()


def centered_filter_spectrum(filters: jax.Array) -> jax.Array:
    """Embed centered compact filters on the fixed FFT grid."""
    if filters.ndim == 3:
        filters = filters[None, ...]
    shape = filters.shape[-3:]
    padding = (
        (0, 0),
        (0, FFT_SIDE - shape[0]),
        (0, FFT_SIDE - shape[1]),
        (0, FFT_SIDE - shape[2]),
    )
    padded = jnp.pad(filters, padding)
    radii = tuple((size - 1) // 2 for size in shape)
    return jnp.fft.fftn(
        jnp.roll(
            padded,
            shift=tuple(-radius for radius in radii),
            axis=(-3, -2, -1),
        ),
        axes=(-3, -2, -1),
    )


def volume_spectrum(volume: jax.Array, whitening_spectrum: jax.Array) -> jax.Array:
    """Pad and whiten one loaded subvolume in Fourier space."""
    spectrum = jnp.fft.fftn(volume, s=(FFT_SIDE,) * 3)
    return spectrum * whitening_spectrum


def accumulate_response(
    score: jax.Array,
    spectrum: jax.Array,
    template_spectrum: jax.Array,
    weight: jax.Array,
) -> jax.Array:
    """Transform one response and accumulate its valid-region energy."""
    response = jnp.fft.ifftn(spectrum * template_spectrum)
    owned = jax.lax.dynamic_slice(
        response,
        (CROP_START,) * 3,
        (OUTPUT_SIDE,) * 3,
    )
    return score + weight * (
        owned.real * owned.real + owned.imag * owned.imag
    )


def score_subvolume_major(
    volumes: jax.Array,
    whitening_filter: jax.Array,
    templates: jax.Array,
    weights: jax.Array,
) -> jax.Array:
    """Reference schedule that rebuilds template spectra per subvolume."""
    whitening_spectrum = centered_filter_spectrum(whitening_filter)[0]
    scores = jnp.zeros(
        (volumes.shape[0], OUTPUT_SIDE, OUTPUT_SIDE, OUTPUT_SIDE),
        dtype=jnp.float32,
    )

    def score_volume(index: jax.Array, all_scores: jax.Array) -> jax.Array:
        spectrum = volume_spectrum(volumes[index], whitening_spectrum)
        score = jnp.zeros((OUTPUT_SIDE,) * 3, dtype=jnp.float32)

        def score_template(
            template_index: jax.Array,
            current: jax.Array,
        ) -> jax.Array:
            template_spectrum = centered_filter_spectrum(
                templates[template_index]
            )[0]
            return accumulate_response(
                current,
                spectrum,
                template_spectrum,
                weights[template_index],
            )

        score = jax.lax.fori_loop(0, templates.shape[0], score_template, score)
        return all_scores.at[index].set(score)

    return jax.lax.fori_loop(0, volumes.shape[0], score_volume, scores)


def score_template_major(
    volumes: jax.Array,
    whitening_filter: jax.Array,
    templates: jax.Array,
    weights: jax.Array,
) -> jax.Array:
    """Reuse every template FFT across all resident subvolumes."""
    whitening_spectrum = centered_filter_spectrum(whitening_filter)[0]
    spectra = jnp.zeros(
        (volumes.shape[0], FFT_SIDE, FFT_SIDE, FFT_SIDE),
        dtype=jnp.complex64,
    )

    def prepare_volume(index: jax.Array, current: jax.Array) -> jax.Array:
        return current.at[index].set(
            volume_spectrum(volumes[index], whitening_spectrum)
        )

    spectra = jax.lax.fori_loop(0, volumes.shape[0], prepare_volume, spectra)
    scores = jnp.zeros(
        (volumes.shape[0], OUTPUT_SIDE, OUTPUT_SIDE, OUTPUT_SIDE),
        dtype=jnp.float32,
    )

    def score_template(
        template_index: jax.Array,
        all_scores: jax.Array,
    ) -> jax.Array:
        template_spectrum = centered_filter_spectrum(
            templates[template_index]
        )[0]

        def score_volume(index: jax.Array, current: jax.Array) -> jax.Array:
            score = accumulate_response(
                current[index],
                spectra[index],
                template_spectrum,
                weights[template_index],
            )
            return current.at[index].set(score)

        return jax.lax.fori_loop(
            0,
            volumes.shape[0],
            score_volume,
            all_scores,
        )

    return jax.lax.fori_loop(0, templates.shape[0], score_template, scores)


def memory_analysis(compiled: Any) -> dict[str, int | None]:
    """Extract the fields exposed by the installed JAX version."""
    analysis = compiled.memory_analysis()
    fields = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "temp_size_in_bytes",
        "alias_size_in_bytes",
    )
    return {field: getattr(analysis, field, None) for field in fields}


def main() -> None:
    """Compile, warm up and time one schedule/tile configuration."""
    args = parse_args()
    if (
        args.tile_size < 1
        or args.template_count < 1
        or args.warmups < 1
        or args.iterations < 1
    ):
        raise ValueError(
            "tile size, template count, warmups and iterations must be positive"
        )
    devices = jax.devices("gpu")
    if len(devices) != 1:
        raise RuntimeError("expose exactly one GPU with CUDA_VISIBLE_DEVICES")
    device = devices[0]
    rng = np.random.default_rng(20260905)
    volumes = jax.device_put(
        rng.standard_normal(
            (args.tile_size, LOADED_SIDE, LOADED_SIDE, LOADED_SIDE),
            dtype=np.float32,
        ),
        device,
    )
    whitening = np.zeros((WHITENING_SIDE,) * 3, dtype=np.float32)
    whitening[(WHITENING_SIDE // 2,) * 3] = 1
    whitening = jax.device_put(whitening, device)
    templates = jax.device_put(
        (
            rng.standard_normal(
                (args.template_count,) + (COMPACT_TEMPLATE_SIDE,) * 3,
                dtype=np.float32,
            )
            + 1j
            * rng.standard_normal(
                (args.template_count,) + (COMPACT_TEMPLATE_SIDE,) * 3,
                dtype=np.float32,
            )
        ).astype(np.complex64),
        device,
    )
    weights = jax.device_put(
        np.linspace(0.75, 1.25, args.template_count, dtype=np.float32),
        device,
    )
    implementation = (
        score_subvolume_major
        if args.schedule == "subvolume-major"
        else score_template_major
    )
    function = jax.jit(implementation, device=device)
    compiled = function.lower(volumes, whitening, templates, weights).compile()
    for _ in range(args.warmups):
        compiled(volumes, whitening, templates, weights).block_until_ready()

    timings = []
    result = None
    for _ in range(args.iterations):
        started = time.perf_counter()
        result = compiled(volumes, whitening, templates, weights)
        result.block_until_ready()
        timings.append(time.perf_counter() - started)
    assert result is not None
    values = np.asarray(result)
    statistics = device.memory_stats() or {}
    median = float(np.median(timings))
    print(
        json.dumps(
            {
                "jax_version": jax.__version__,
                "device": device.device_kind,
                "schedule": args.schedule,
                "tile_size": args.tile_size,
                "template_count": args.template_count,
                "median_seconds": median,
                "median_seconds_per_subvolume": median / args.tile_size,
                "template_subvolume_pairs_per_second": (
                    args.tile_size * args.template_count / median
                ),
                "compiled_memory": memory_analysis(compiled),
                "runtime_bytes_in_use": statistics.get("bytes_in_use"),
                "runtime_peak_bytes_in_use": statistics.get("peak_bytes_in_use"),
                "score_checksum": float(np.sum(values, dtype=np.float64)),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
