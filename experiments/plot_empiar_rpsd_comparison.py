#!/usr/bin/env python3
"""Plot initial and whitened patch-RPSD summaries across EMPIAR-10045 tomograms."""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


DEFAULT_RESULTS = {
    "05": Path("results/empiar-10045-tomo05-trace58"),
    "08": Path("results/empiar-10045-trace58-v2"),
    "09": Path("results/empiar-10045-tomo09-trace58"),
    "10": Path("results/empiar-10045-tomo10-trace58"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/empiar-10045-rpsd-comparison.png"),
    )
    return parser.parse_args()


def load_summary(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    with path.open("rb") as stream:
        result = pickle.load(stream)
    rpsds = np.asarray(result.rpsds, dtype=np.float64)
    if rpsds.ndim != 2 or not np.all(np.isfinite(rpsds)):
        raise ValueError(f"invalid RPSD array in {path}")
    median = np.median(rpsds, axis=0)
    lower, upper = np.percentile(rpsds, [25, 75], axis=0)
    return np.asarray(result.radial_points), median, np.stack((lower, upper)), len(rpsds)


def normalize_shape(frequency: np.ndarray, spectrum: np.ndarray) -> np.ndarray:
    interior = (frequency >= 0.10) & (frequency <= 0.90)
    scale = np.median(spectrum[interior])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("RPSD normalization scale must be positive and finite")
    return spectrum / scale


def main() -> None:
    args = parse_args()
    stages = {
        "Initial band-passed patch RPSD": "01_initial_patch_rpsds.pkl",
        "After whitening": "04_whitened_patch_rpsds.pkl",
    }
    summaries: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]]] = {}
    reference_frequency: np.ndarray | None = None
    for title, filename in stages.items():
        summaries[title] = {}
        for tomogram, directory in DEFAULT_RESULTS.items():
            summary = load_summary(directory / filename)
            frequency = summary[0] / np.pi
            if reference_frequency is None:
                reference_frequency = frequency
            elif not np.array_equal(reference_frequency, frequency):
                raise ValueError("tomograms do not share the same radial grid")
            summaries[title][tomogram] = (frequency, *summary[1:])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    colors = dict(zip(DEFAULT_RESULTS, plt.get_cmap("tab10").colors, strict=False))
    rows: list[list[object]] = []
    for row, (title, stage) in enumerate(summaries.items()):
        raw_axis, normalized_axis = axes[row]
        for tomogram, (frequency, median, quartiles, count) in stage.items():
            color = colors[tomogram]
            label = f"Tomo {tomogram} ({count:,} patches)"
            raw_axis.plot(frequency, median, color=color, linewidth=1.8, label=label)
            raw_axis.fill_between(
                frequency,
                quartiles[0],
                quartiles[1],
                color=color,
                alpha=0.08,
                linewidth=0,
            )
            normalized = normalize_shape(frequency, median)
            normalized_axis.plot(
                frequency, normalized, color=color, linewidth=1.8, label=label
            )
            for radial_frequency, raw_value, normalized_value in zip(
                frequency, median, normalized, strict=True
            ):
                rows.append(
                    [title, tomogram, count, radial_frequency, raw_value, normalized_value]
                )

        raw_axis.set_title(f"{title}: absolute scale")
        raw_axis.set_yscale("log")
        raw_axis.set_ylabel("Median power")
        raw_axis.grid(alpha=0.25)
        raw_axis.legend(fontsize=8)
        normalized_axis.set_title(f"{title}: normalized shape")
        normalized_axis.set_yscale("log")
        normalized_axis.set_ylabel("Relative median power")
        normalized_axis.grid(alpha=0.25)

    for axis in axes[-1]:
        axis.set_xlabel("Radial spatial frequency / Nyquist")
    fig.suptitle("EMPIAR-10045 patch RPSD comparison", fontsize=15)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["stage", "tomogram", "patch_count", "frequency_over_nyquist", "median_rpsd", "normalized_median_rpsd"]
        )
        writer.writerows(rows)
    print(args.output)
    print(csv_path)


if __name__ == "__main__":
    main()
