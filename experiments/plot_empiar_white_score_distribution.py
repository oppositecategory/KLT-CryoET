#!/usr/bin/env python3
"""Plot score separation for the original scalar-noise EMPIAR picker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/empiar-10045-trace58-v2"),
    )
    return parser.parse_args()


def load_truth(path: Path) -> np.ndarray:
    values = np.loadtxt(path, ndmin=2)
    # Deposited coordinates are x,y,z; detector coordinates are z,y,x.
    return np.asarray(values[:, :3][:, ::-1], dtype=np.float64)


def maximum_matches(
    particles: np.ndarray, truth: np.ndarray, radius: float
) -> tuple[int, np.ndarray]:
    distances = cdist(particles[:, :3], truth)
    within = distances <= radius
    prediction_indices, truth_indices = linear_sum_assignment(~within)
    accepted = within[prediction_indices, truth_indices]
    return int(np.count_nonzero(accepted)), prediction_indices[accepted]


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    with (results_dir / "00_manifest.json").open() as stream:
        manifest = json.load(stream)
    particles = np.load(
        results_dir / "08_positive_particles_top4096_zyx.npy",
        allow_pickle=False,
    )
    truth = load_truth(Path(manifest["ground_truth"]))
    radius = float(manifest["match_radius_voxels"])

    matched_count, matched_indices = maximum_matches(particles, truth, radius)
    matched_mask = np.zeros(len(particles), dtype=bool)
    matched_mask[matched_indices] = True

    sample_counts = np.unique(
        np.r_[
            np.arange(50, len(particles) + 1, 100),
            [158, 454, 1000, 2000, 3000, 4000, 5000, len(particles)],
        ]
    )
    sample_counts = sample_counts[sample_counts <= len(particles)]
    recalls = np.empty(sample_counts.size)
    precisions = np.empty(sample_counts.size)
    matches = np.empty(sample_counts.size, dtype=int)
    for index, count in enumerate(sample_counts):
        matches[index], _ = maximum_matches(particles[:count], truth, radius)
        recalls[index] = matches[index] / len(truth)
        precisions[index] = matches[index] / count

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    log_scores = np.log10(particles[:, 3])
    bins = np.linspace(log_scores.min(), log_scores.max(), 55)
    axes[0].hist(
        log_scores[~matched_mask], bins=bins, density=True, alpha=0.7,
        label=f"Unmatched ({np.count_nonzero(~matched_mask):,})",
    )
    axes[0].hist(
        log_scores[matched_mask], bins=bins, density=True, alpha=0.7,
        label=f"Matched ({matched_count:,})",
    )
    axes[0].set_xlabel(r"$\log_{10}$(raw likelihood score)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Score distributions overlap strongly")
    axes[0].legend()

    rank_edges = np.arange(0, len(particles) + 500, 500)
    rank_edges[-1] = len(particles)
    centers = []
    fractions = []
    for start, stop in zip(rank_edges[:-1], rank_edges[1:]):
        if stop <= start:
            continue
        centers.append((start + stop) / 2)
        fractions.append(np.mean(matched_mask[start:stop]))
    axes[1].bar(centers, fractions, width=450)
    axes[1].axhline(
        matched_count / len(particles), color="black", linestyle="--",
        label="Overall precision",
    )
    axes[1].set_xlabel("Score rank (highest first)")
    axes[1].set_ylabel("Matched fraction in rank bin")
    axes[1].set_title("Extreme scores are artifact-enriched")
    axes[1].legend()

    axes[2].plot(sample_counts, recalls, label="Recall")
    axes[2].plot(sample_counts, precisions, label="Precision")
    axes[2].scatter([454, 4000, len(particles)], [
        recalls[np.argmin(np.abs(sample_counts - 454))],
        recalls[np.argmin(np.abs(sample_counts - 4000))],
        recalls[-1],
    ], s=28)
    axes[2].set_xlabel("Number of highest-scoring post-NMS picks retained")
    axes[2].set_ylabel("Fraction")
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Threshold sweep")
    axes[2].legend()

    figure.suptitle(
        "EMPIAR-10045 Tomogram 08 — scalar-noise 58%-trace score diagnostics"
    )
    figure.tight_layout()
    output = results_dir / "white_score_distribution.png"
    figure.savefig(output, dpi=180, bbox_inches="tight")

    table = np.column_stack((sample_counts, particles[sample_counts - 1, 3], matches,
                             recalls, precisions))
    np.savetxt(
        results_dir / "white_score_threshold_sweep.csv",
        table,
        delimiter=",",
        header="retained_picks,score_threshold,matches,recall,precision",
        comments="",
        fmt=["%d", "%.8g", "%d", "%.8g", "%.8g"],
    )
    print(output)


if __name__ == "__main__":
    main()
