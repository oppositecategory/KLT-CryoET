#!/usr/bin/env python3
"""Analyze ALS density, patch variance, and scalar KLT score relationships."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-klt-als")

import importlib.util
import json
import pickle
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.stats import rankdata, spearmanr

from kltpicker_3d.alt_least_squares import alternating_least_squares_solver


ROOT = Path(__file__).resolve().parents[1]
RUNS = (
    ("05", ROOT / "results/empiar-10045-tomo05-trace58"),
    ("08", ROOT / "results/empiar-10045-trace58-v2"),
    ("09", ROOT / "results/empiar-10045-tomo09-trace58"),
    ("10", ROOT / "results/empiar-10045-tomo10-trace58"),
)
OUTPUT = ROOT / "results/empiar-10045-score-analysis"


def load_experiment_module():
    path = ROOT / "experiments/EMPIAR-10045.py"
    spec = importlib.util.spec_from_file_location("empiar_experiment", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def auc(labels: np.ndarray, values: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    ranks = rankdata(values)
    positive_count = int(np.count_nonzero(labels))
    negative_count = labels.size - positive_count
    rank_sum = float(np.sum(ranks[labels]))
    return (
        rank_sum - positive_count * (positive_count + 1) / 2
    ) / (positive_count * negative_count)


def main() -> None:
    experiment = load_experiment_module()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = []
    rows = []

    for tomo, directory in RUNS:
        with (directory / "00_manifest.json").open() as stream:
            manifest = json.load(stream)
        with (directory / "04_whitened_patch_rpsds.pkl").open("rb") as stream:
            extraction = pickle.load(stream)
        factorization = alternating_least_squares_solver(
            jnp.asarray(extraction.rpsds), 500, 1e-4
        )
        alpha_grid = np.asarray(factorization.alpha).reshape(
            extraction.patch_grid_shape
        )
        variance_grid = np.asarray(extraction.variances).reshape(
            extraction.patch_grid_shape
        )

        particle_path = next(
            directory.glob("08_positive_particles_top*_zyx.npy")
        )
        particles = np.load(particle_path, allow_pickle=False)
        truth = experiment.load_ground_truth(Path(manifest["ground_truth"]))
        _, matches = experiment.evaluate_recall(
            particles, truth, float(manifest["match_radius_voxels"])
        )
        matched = np.zeros(len(particles), dtype=bool)
        matched[matches[:, 0].astype(int)] = True

        # RPSD values live at centers of non-overlapping patches. Interpolate
        # their coarse fields at every candidate center.
        coordinates = (
            particles[:, :3] / float(extraction.patch_size) - 0.5
        ).T
        alpha = map_coordinates(
            alpha_grid, coordinates, order=1, mode="nearest"
        )
        variance = map_coordinates(
            variance_grid, coordinates, order=1, mode="nearest"
        )
        score = particles[:, 3]
        patch_alpha = alpha_grid.ravel()
        patch_variance = variance_grid.ravel()
        records.append(
            (tomo, patch_alpha, patch_variance, alpha, variance, score, matched)
        )
        rows.append(
            (
                tomo,
                len(particles),
                int(np.count_nonzero(matched)),
                spearmanr(patch_alpha, patch_variance).statistic,
                spearmanr(alpha, score).statistic,
                spearmanr(variance, score).statistic,
                auc(matched, alpha),
                auc(matched, variance),
                auc(matched, score),
                np.median(alpha[matched]),
                np.median(alpha[~matched]),
                np.median(variance[matched]),
                np.median(variance[~matched]),
            )
        )

    figure, axes = plt.subplots(4, 3, figsize=(16, 18))
    for row_index, record in enumerate(records):
        tomo, patch_alpha, patch_variance, alpha, variance, score, matched = record
        alpha_floor = max(np.quantile(patch_alpha[patch_alpha > 0], 0.01), 1e-8)

        axis = axes[row_index, 0]
        axis.hexbin(
            np.log10(np.maximum(patch_alpha, alpha_floor)),
            patch_variance,
            gridsize=55,
            bins="log",
            mincnt=1,
            cmap="viridis",
        )
        rho = spearmanr(patch_alpha, patch_variance).statistic
        axis.set_title(f"Tomo {tomo}: all RPSD patches, Spearman={rho:.3f}")
        axis.set_xlabel(r"$\log_{10}(\alpha)$")
        axis.set_ylabel("Whitened patch variance")

        axis = axes[row_index, 1]
        axis.hexbin(
            np.log10(np.maximum(alpha[~matched], alpha_floor)),
            np.log10(score[~matched]),
            gridsize=55,
            bins="log",
            mincnt=1,
            cmap="Blues",
        )
        axis.scatter(
            np.log10(np.maximum(alpha[matched], alpha_floor)),
            np.log10(score[matched]),
            s=7,
            alpha=0.5,
            color="tab:orange",
            label="Matched",
        )
        rho = spearmanr(alpha, score).statistic
        axis.set_title(f"Candidate score vs. alpha, Spearman={rho:.3f}")
        axis.set_xlabel(r"Interpolated $\log_{10}(\alpha)$")
        axis.set_ylabel(r"$\log_{10}$(raw score)")
        axis.legend()

        axis = axes[row_index, 2]
        axis.hexbin(
            variance[~matched],
            np.log10(score[~matched]),
            gridsize=55,
            bins="log",
            mincnt=1,
            cmap="Blues",
        )
        axis.scatter(
            variance[matched],
            np.log10(score[matched]),
            s=7,
            alpha=0.5,
            color="tab:orange",
            label="Matched",
        )
        rho = spearmanr(variance, score).statistic
        axis.set_title(f"Candidate score vs. variance, Spearman={rho:.3f}")
        axis.set_xlabel("Interpolated whitened variance")
        axis.set_ylabel(r"$\log_{10}$(raw score)")
        axis.legend()

    figure.suptitle(
        "EMPIAR-10045: whitened ALS density, local variance, and KLT score",
        fontsize=16,
    )
    figure.tight_layout()
    figure.savefig(OUTPUT / "als_score_correlations_by_tomogram.png", dpi=180)
    figure.savefig(OUTPUT / "als_score_correlations_by_tomogram.pdf")

    header = (
        "tomogram,picks,matches,patch_alpha_variance_spearman,"
        "candidate_alpha_score_spearman,candidate_variance_score_spearman,"
        "alpha_match_auc,variance_match_auc,score_match_auc,"
        "matched_alpha_median,unmatched_alpha_median,"
        "matched_variance_median,unmatched_variance_median"
    )
    np.savetxt(
        OUTPUT / "als_score_correlations.csv",
        np.asarray(rows, dtype=object),
        delimiter=",",
        header=header,
        comments="",
        fmt=["%s", "%d", "%d"] + ["%.8g"] * 10,
    )
    print(OUTPUT / "als_score_correlations_by_tomogram.png")


if __name__ == "__main__":
    main()
