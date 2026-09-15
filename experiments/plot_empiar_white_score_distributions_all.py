#!/usr/bin/env python3
"""Plot scalar-noise matched/unmatched score distributions across tomograms."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


ROOT = Path(__file__).resolve().parents[1]
RUNS = (
    ("Tomogram 05", ROOT / "results/empiar-10045-tomo05-trace58"),
    ("Tomogram 08", ROOT / "results/empiar-10045-trace58-v2"),
    ("Tomogram 09", ROOT / "results/empiar-10045-tomo09-trace58"),
    ("Tomogram 10", ROOT / "results/empiar-10045-tomo10-trace58"),
)
OUTPUT_DIR = ROOT / "results/empiar-10045-score-analysis"


def load_truth(path: Path) -> np.ndarray:
    coordinates = np.loadtxt(path, ndmin=2)
    return np.asarray(coordinates[:, :3][:, ::-1], dtype=np.float64)


def matched_mask(
    particles: np.ndarray, truth: np.ndarray, radius: float
) -> np.ndarray:
    within = cdist(particles[:, :3], truth) <= radius
    prediction_indices, truth_indices = linear_sum_assignment(~within)
    accepted = within[prediction_indices, truth_indices]
    mask = np.zeros(len(particles), dtype=bool)
    mask[prediction_indices[accepted]] = True
    return mask


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records: list[tuple[str, np.ndarray, np.ndarray]] = []
    summary_rows = []
    for label, directory in RUNS:
        with (directory / "00_manifest.json").open() as stream:
            manifest = json.load(stream)
        particle_path = next(directory.glob("08_positive_particles_top*_zyx.npy"))
        particles = np.load(particle_path, allow_pickle=False)
        truth = load_truth(Path(manifest["ground_truth"]))
        mask = matched_mask(
            particles, truth, float(manifest["match_radius_voxels"])
        )
        matched_scores = particles[mask, 3]
        unmatched_scores = particles[~mask, 3]
        records.append((label, matched_scores, unmatched_scores))
        quantiles = np.quantile(matched_scores, [0, 0.01, 0.1, 0.5, 0.9, 0.99, 1])
        summary_rows.append(
            [label, len(particles), len(matched_scores), *quantiles]
        )

    all_scores = np.concatenate(
        [np.r_[matched, unmatched] for _, matched, unmatched in records]
    )
    log_range = np.log10([all_scores.min(), all_scores.max()])
    bins = np.linspace(log_range[0], log_range[1], 60)
    figure, axes = plt.subplots(2, 3, figsize=(17, 9))

    for axis, (label, matched, unmatched) in zip(axes.flat[:4], records):
        axis.hist(
            np.log10(unmatched), bins=bins, density=True, alpha=0.65,
            label=f"Unmatched ({len(unmatched):,})",
        )
        axis.hist(
            np.log10(matched), bins=bins, density=True, alpha=0.72,
            label=f"Matched ({len(matched):,})",
        )
        axis.set_title(
            f"{label}\nmatched min={matched.min():.1f}, median={np.median(matched):.1f}"
        )
        axis.set_xlabel(r"$\log_{10}$(raw likelihood score)")
        axis.set_ylabel("Density")
        axis.legend(fontsize=9)

    aggregate_matched = np.concatenate([record[1] for record in records])
    aggregate_unmatched = np.concatenate([record[2] for record in records])
    axis = axes.flat[4]
    axis.hist(
        np.log10(aggregate_unmatched), bins=bins, density=True, alpha=0.65,
        label=f"Unmatched ({len(aggregate_unmatched):,})",
    )
    axis.hist(
        np.log10(aggregate_matched), bins=bins, density=True, alpha=0.72,
        label=f"Matched ({len(aggregate_matched):,})",
    )
    axis.set_title(
        "All four tomograms\n"
        f"matched min={aggregate_matched.min():.1f}, "
        f"median={np.median(aggregate_matched):.1f}"
    )
    axis.set_xlabel(r"$\log_{10}$(raw likelihood score)")
    axis.set_ylabel("Density")
    axis.legend(fontsize=9)

    axis = axes.flat[5]
    for label, matched, _ in records:
        ordered = np.sort(matched)
        cumulative = np.arange(1, len(ordered) + 1) / len(ordered)
        axis.plot(ordered, cumulative, label=label)
    axis.set_xscale("log")
    axis.set_xlabel("Raw likelihood score (log scale)")
    axis.set_ylabel("Fraction of matched particles at or below score")
    axis.set_title("Matched-score empirical CDF\n(distance from zero is tomogram-dependent)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=9)

    figure.suptitle(
        "EMPIAR-10045 scalar-noise 58%-trace scores after positive-score NMS",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "white_score_distributions_all.png", dpi=180)
    figure.savefig(OUTPUT_DIR / "white_score_distributions_all.pdf")

    with (OUTPUT_DIR / "white_matched_score_summary.csv").open("w") as stream:
        stream.write(
            "tomogram,picks,matches,min,p01,p10,median,p90,p99,max\n"
        )
        for row in summary_rows:
            stream.write(
                f"{row[0]},{row[1]},{row[2]},"
                + ",".join(f"{value:.8g}" for value in row[3:])
                + "\n"
            )
    print(OUTPUT_DIR / "white_score_distributions_all.png")


if __name__ == "__main__":
    main()
