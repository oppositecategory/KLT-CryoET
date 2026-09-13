#!/usr/bin/env python3
"""Create an interpretable FSC report from a RELION postprocess STAR file."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("star", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unmasked-resolution", type=float, default=None)
    parser.add_argument("--mask-occupancy", type=float, default=None)
    return parser.parse_args()


def read_postprocess_star(path: Path) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    lines = path.read_text().splitlines()
    scalars: dict[str, float] = {}
    for line in lines:
        fields = line.split()
        if len(fields) == 2 and fields[0].startswith("_rln"):
            try:
                scalars[fields[0]] = float(fields[1])
            except ValueError:
                pass

    start = lines.index("data_fsc")
    labels: list[str] = []
    rows: list[list[float]] = []
    reading_rows = False
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if stripped.startswith("data_") and rows:
            break
        if stripped.startswith("_rln"):
            labels.append(stripped.split()[0])
            continue
        if labels and stripped and not stripped.startswith(("#", "loop_")):
            fields = stripped.split()
            if len(fields) == len(labels):
                try:
                    rows.append([float(value) for value in fields])
                    reading_rows = True
                    continue
                except ValueError:
                    pass
        if reading_rows and not stripped:
            break
    values = np.asarray(rows, dtype=float)
    return scalars, {label: values[:, i] for i, label in enumerate(labels)}


def crossing_frequency(frequency: np.ndarray, values: np.ndarray, level: float) -> float:
    for i in range(1, len(values)):
        if values[i - 1] >= level and values[i] < level:
            fraction = (level - values[i - 1]) / (values[i] - values[i - 1])
            return float(frequency[i - 1] + fraction * (frequency[i] - frequency[i - 1]))
    return float("nan")


def add_resolution_axis(axis: plt.Axes) -> None:
    def reciprocal(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        return np.divide(1.0, values, out=np.full_like(values, np.inf), where=values > 0)

    top = axis.secondary_xaxis(
        "top",
        functions=(reciprocal, reciprocal),
    )
    top.set_xlabel("Resolution (Å; smaller is finer)")
    top.set_xticks([100, 50, 30, 20, 17, 15, 14, 13, 12])


def main() -> None:
    args = parse_args()
    scalars, columns = read_postprocess_star(args.star)
    frequency = columns["_rlnResolution"][1:]
    corrected = columns["_rlnFourierShellCorrelationCorrected"][1:]
    unmasked = columns["_rlnFourierShellCorrelationUnmaskedMaps"][1:]
    masked = columns["_rlnFourierShellCorrelationMaskedMaps"][1:]
    randomized = columns["_rlnCorrectedFourierShellCorrelationPhaseRandomizedMaskedMaps"][1:]

    threshold = 0.143
    interpolated_frequency = crossing_frequency(frequency, corrected, threshold)
    interpolated_resolution = 1.0 / interpolated_frequency
    shell_resolution = scalars["_rlnFinalResolution"]
    bfactor = scalars["_rlnBfactorUsedForSharpening"]
    nyquist = 1.0 / frequency[-1]

    colors = {
        "corrected": "#1769aa",
        "unmasked": "#343a40",
        "masked": "#ef6c00",
        "randomized": "#b71c1c",
    }
    fig, (full, zoom) = plt.subplots(
        1, 2, figsize=(14.2, 6.4), gridspec_kw={"width_ratios": [1.45, 1]}
    )

    for axis in (full, zoom):
        axis.plot(frequency, corrected, color=colors["corrected"], lw=2.8, label="Corrected masked FSC")
        axis.plot(frequency, unmasked, color=colors["unmasked"], lw=1.8, label="Unmasked half-map FSC")
        axis.plot(frequency, masked, color=colors["masked"], lw=1.6, ls="--", label="Raw masked FSC")
        axis.plot(frequency, randomized, color=colors["randomized"], lw=1.4, ls=":", label="Phase-randomized control")
        axis.axhline(threshold, color="#6a1b9a", lw=1.6, ls="--", label="0.143 criterion")
        axis.axvline(interpolated_frequency, color=colors["corrected"], lw=1.2, ls=":")
        axis.axvline(1.0 / nyquist, color="#777777", lw=1.1, ls="-.")
        axis.grid(True, color="#d9dee3", lw=0.7, alpha=0.8)
        axis.set_xlabel("Spatial frequency (Å⁻¹; farther right is finer detail)")
        axis.set_ylabel("Fourier shell correlation")
        axis.spines[["top", "right"]].set_visible(False)

    full.set_xlim(frequency[0] * 0.75, frequency[-1] * 1.015)
    full.set_ylim(-0.08, 1.04)
    add_resolution_axis(full)
    full.legend(loc="lower left", frameon=True, framealpha=0.96, fontsize=9)

    zoom.set_xlim(1.0 / 32.0, frequency[-1] * 1.015)
    zoom.set_ylim(-0.04, 0.82)
    add_resolution_axis(zoom)
    zoom.annotate(
        f"Corrected 0.143 crossing\n{shell_resolution:.2f} Å shell\n{interpolated_resolution:.2f} Å interpolated",
        xy=(interpolated_frequency, threshold),
        xytext=(0.058, 0.55),
        arrowprops={"arrowstyle": "->", "color": colors["corrected"], "lw": 1.5},
        bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": colors["corrected"], "alpha": 0.95},
        fontsize=10,
    )
    if args.unmasked_resolution is not None:
        zoom.axvline(1.0 / args.unmasked_resolution, color=colors["unmasked"], lw=1.1, ls=":")
        zoom.text(
            1.0 / args.unmasked_resolution,
            0.74,
            f" unmasked\n {args.unmasked_resolution:.2f} Å",
            color=colors["unmasked"],
            va="top",
            fontsize=9,
        )
    zoom.text(1.0 / nyquist, 0.74, f" Nyquist\n {nyquist:.2f} Å", color="#555555", va="top", fontsize=9)

    occupancy = ""
    if args.mask_occupancy is not None:
        occupancy = f"  •  soft-mask occupancy {100 * args.mask_occupancy:.1f}%"
    fig.suptitle("RELION gold-standard FSC — four-tomogram class-4 refinement", fontsize=16, weight="bold")
    fig.text(
        0.5,
        0.015,
        f"Corrected resolution {shell_resolution:.2f} Å ({interpolated_resolution:.2f} Å interpolated)"
        f"  •  unmasked {args.unmasked_resolution:.2f} Å"
        f"  •  Nyquist {nyquist:.2f} Å  •  sharpening B = {bfactor:.1f} Å²{occupancy}",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Wrote {args.output.with_suffix('.png')}")
    print(f"Wrote {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
