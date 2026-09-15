#!/usr/bin/env python3
"""Compare ALS-estimated particle/signal RPSDs across EMPIAR-10045 tomograms."""

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
        default=Path("results/empiar-10045-als-signal-rpsd-comparison.png"),
    )
    return parser.parse_args()


def load_pickle(path: Path) -> object:
    with path.open("rb") as stream:
        return pickle.load(stream)


def normalized_positive_shape(
    frequency: np.ndarray, spectrum: np.ndarray
) -> np.ndarray:
    del frequency
    positive = np.isfinite(spectrum) & (spectrum > 0)
    scale = np.max(spectrum[positive]) if np.any(positive) else np.nan
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("ALS signal RPSD has no positive interior support")
    return spectrum / scale


def main() -> None:
    args = parse_args()
    stages = {
        "Initial ALS signal RPSD": ("01_initial_patch_rpsds.pkl", "02_initial_als.pkl"),
        "Whitened ALS signal RPSD used for Fredholm": (
            "04_whitened_patch_rpsds.pkl",
            "05_whitened_als.pkl",
        ),
    }
    colors = dict(zip(DEFAULT_RESULTS, plt.get_cmap("tab10").colors, strict=False))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    csv_rows: list[list[object]] = []
    reference_frequency: np.ndarray | None = None

    for row, (title, (extraction_name, model_name)) in enumerate(stages.items()):
        absolute_axis, normalized_axis = axes[row]
        for tomogram, directory in DEFAULT_RESULTS.items():
            extraction = load_pickle(directory / extraction_name)
            model = load_pickle(directory / model_name)
            frequency = np.asarray(extraction.radial_points, dtype=np.float64) / np.pi
            signal = np.asarray(model.particle_psd, dtype=np.float64)
            noise = np.asarray(model.noise_psd, dtype=np.float64)
            if signal.shape != frequency.shape or noise.shape != frequency.shape:
                raise ValueError(f"ALS/grid shape mismatch for tomogram {tomogram}")
            if reference_frequency is None:
                reference_frequency = frequency
            elif not np.array_equal(reference_frequency, frequency):
                raise ValueError("tomograms do not share the same radial grid")

            positive = signal > 0
            normalized = normalized_positive_shape(frequency, signal)
            color = colors[tomogram]
            label = f"Tomo {tomogram}"
            absolute_axis.plot(
                frequency[positive], signal[positive], color=color, linewidth=1.9, label=label
            )
            normalized_axis.plot(
                frequency[positive],
                normalized[positive],
                color=color,
                linewidth=1.9,
                label=label,
            )
            for radial_frequency, signal_value, noise_value, normalized_value in zip(
                frequency, signal, noise, normalized, strict=True
            ):
                csv_rows.append(
                    [
                        title,
                        tomogram,
                        radial_frequency,
                        signal_value,
                        noise_value,
                        signal_value / noise_value if noise_value > 0 else np.nan,
                        normalized_value,
                    ]
                )

        absolute_axis.set_title(f"{title}: absolute scale")
        absolute_axis.set_yscale("log")
        absolute_axis.set_ylabel("ALS particle/signal power")
        absolute_axis.grid(alpha=0.25)
        absolute_axis.legend()
        normalized_axis.set_title(f"{title}: normalized shape")
        normalized_axis.set_yscale("log")
        normalized_axis.set_ylabel("Relative signal power")
        normalized_axis.grid(alpha=0.25)

    for axis in axes[-1]:
        axis.set_xlabel("Radial spatial frequency / Nyquist")
    fig.suptitle("EMPIAR-10045 ALS-estimated signal RPSD", fontsize=15)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "stage",
                "tomogram",
                "frequency_over_nyquist",
                "particle_psd",
                "noise_psd",
                "particle_to_noise_psd",
                "normalized_particle_psd",
            ]
        )
        writer.writerows(csv_rows)
    print(args.output)
    print(csv_path)


if __name__ == "__main__":
    main()
