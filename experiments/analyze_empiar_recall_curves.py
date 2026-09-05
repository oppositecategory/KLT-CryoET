"""Generate full-tomogram recall-versus-pick-budget curves from checkpoints."""

from __future__ import annotations

import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


RESULTS_ROOT = REPOSITORY_ROOT / "results"
OUTPUT_DIRECTORY = RESULTS_ROOT / "empiar-10045-recall-curves"
TRUTH_PATH = (
    Path("/data/yoelsh/datasets/10045/pristine/data/ribosomes")
    / "AnticipatedResults/Tomograms/08/IS002_291013_008.coords"
)
PARTICLE_COUNT = 454
MATCH_RADIUS_VOXELS = 59.30424910189675
MAXIMUM_FACTOR = 10


@dataclass(frozen=True)
class Run:
    """One saved full-tomogram scoring run."""

    key: str
    label: str
    candidates_path: Path
    saved_particles_path: Path
    color: str
    initial_pool_size: int


RUNS = (
    Run(
        key="independent_normalization",
        label="Earlier scoring (independent normalization)",
        candidates_path=RESULTS_ROOT / "empiar-10045/07_candidates.npy",
        saved_particles_path=RESULTS_ROOT / "empiar-10045/08_particles_zyx.npy",
        color="#4477AA",
        initial_pool_size=3_200_000,
    ),
    Run(
        key="bandpass_block_qr",
        label="5% band-pass + $(\\ell,m)$ block QR",
        candidates_path=(
            RESULTS_ROOT
            / "empiar-10045-bandpass-block-qr/07_candidates_top4096.npy"
        ),
        saved_particles_path=(
            RESULTS_ROOT
            / "empiar-10045-bandpass-block-qr/08_particles_top4096_zyx.npy"
        ),
        color="#CC6677",
        initial_pool_size=200_000,
    ),
)


def configure_logging() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=(
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(OUTPUT_DIRECTORY / "analysis.log", mode="w"),
        ),
    )


def exact_top_pool_order(
    candidates: npt.NDArray[np.float64],
    requested_pool_size: int,
) -> npt.NDArray[np.int64]:
    """Return exact score/coordinate order for the highest-scoring pool."""
    scores = candidates[:, 3]
    finite_scores = np.isfinite(scores)
    finite_count = int(np.count_nonzero(finite_scores))
    if not finite_count:
        return np.empty(0, dtype=np.int64)
    pool_size = min(requested_pool_size, finite_count)
    if pool_size == finite_count:
        indices = np.flatnonzero(finite_scores)
    else:
        finite_indices = np.flatnonzero(finite_scores)
        finite_values = np.asarray(scores[finite_indices])
        selected = np.argpartition(finite_values, -pool_size)[-pool_size:]
        cutoff = float(np.min(finite_values[selected]))
        # Include every cutoff tie so omission cannot change lexicographic order.
        indices = finite_indices[finite_values >= cutoff]
    rows = np.asarray(candidates[indices])
    row_finite = np.all(np.isfinite(rows), axis=1)
    indices = indices[row_finite]
    rows = rows[row_finite]
    order = np.lexsort((rows[:, 2], rows[:, 1], rows[:, 0], -rows[:, 3]))
    return np.asarray(indices[order], dtype=np.int64)


def spatial_hash_nms(
    candidates: npt.NDArray[np.float64],
    ordered_indices: npt.NDArray[np.int64],
    *,
    radius: float,
    max_picks: int,
) -> npt.NDArray[np.float64]:
    """Apply exact greedy spherical NMS without quadratic all-pairs scans."""
    accepted = np.empty((max_picks, 4), dtype=np.float64)
    cells: dict[tuple[int, int, int], list[int]] = {}
    count = 0
    radius_squared = radius**2
    for candidate_index in ordered_indices:
        candidate = np.asarray(candidates[candidate_index], dtype=np.float64)
        cell = tuple(np.floor(candidate[:3] / radius).astype(np.int64))
        reject = False
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    neighbor = (cell[0] + dz, cell[1] + dy, cell[2] + dx)
                    indices = cells.get(neighbor)
                    if indices is None:
                        continue
                    differences = accepted[indices, :3] - candidate[:3]
                    if np.any(np.sum(differences * differences, axis=1) <= radius_squared):
                        reject = True
                        break
                if reject:
                    break
            if reject:
                break
        if reject:
            continue
        accepted[count] = candidate
        cells.setdefault(cell, []).append(count)
        count += 1
        if count == max_picks:
            break
    return accepted[:count].copy()


def extended_nms(
    candidates: npt.NDArray[np.float64],
    *,
    radius: float,
    max_picks: int,
    initial_pool_size: int,
) -> tuple[npt.NDArray[np.float64], int]:
    """Expand a top-score pool until it proves the requested NMS prefix."""
    pool_size = min(
        len(candidates),
        max(initial_pool_size, 20 * max_picks),
    )
    while True:
        logging.info("NMS trial with exact top-score pool of %s rows", f"{pool_size:,}")
        order = exact_top_pool_order(candidates, pool_size)
        accepted = spatial_hash_nms(
            candidates,
            order,
            radius=radius,
            max_picks=max_picks,
        )
        logging.info("NMS accepted %s/%s requested picks", f"{len(accepted):,}", f"{max_picks:,}")
        if len(accepted) == max_picks or pool_size == len(candidates):
            return accepted, len(order)
        pool_size = min(len(candidates), 2 * pool_size)


def validate_saved_prefix(
    observed: npt.NDArray[np.float64],
    saved_path: Path,
) -> None:
    """Ensure the accelerated NMS reproduces the historical top-454 output."""
    expected = np.load(saved_path)
    if len(observed) < len(expected):
        raise AssertionError("extended NMS returned fewer rows than the saved result")
    actual = observed[: len(expected)]
    if not np.array_equal(actual[:, :3], expected[:, :3]):
        mismatch = int(np.flatnonzero(np.any(actual[:, :3] != expected[:, :3], axis=1))[0])
        raise AssertionError(f"saved NMS coordinate mismatch at rank {mismatch + 1}")
    # Historical particle checkpoints normalize scores by the leading pick,
    # whereas stage-7 candidates retain the raw likelihood scale.
    actual_scores = actual[:, 3] / actual[0, 3]
    if not np.allclose(actual_scores, expected[:, 3], rtol=2e-6, atol=2e-7):
        maximum_error = float(np.max(np.abs(actual_scores - expected[:, 3])))
        raise AssertionError(
            f"saved normalized NMS scores do not match (max error={maximum_error})"
        )


def incremental_recall_curve(
    predictions: npt.NDArray[np.float64],
    truth_zyx: npt.NDArray[np.float64],
    radius: float,
) -> npt.NDArray[np.int64]:
    """Return exact maximum-cardinality matches for every ranked prefix."""
    truth_tree = cKDTree(truth_zyx)
    adjacency: list[list[int]] = []
    truth_to_prediction = np.full(len(truth_zyx), -1, dtype=np.int64)
    matched_counts = np.empty(len(predictions), dtype=np.int64)
    matched_count = 0

    def augment(prediction_index: int, visited_truth: npt.NDArray[np.bool_]) -> bool:
        for truth_index in adjacency[prediction_index]:
            if visited_truth[truth_index]:
                continue
            visited_truth[truth_index] = True
            previous = int(truth_to_prediction[truth_index])
            if previous < 0 or augment(previous, visited_truth):
                truth_to_prediction[truth_index] = prediction_index
                return True
        return False

    for prediction_index, coordinate in enumerate(predictions[:, :3]):
        adjacency.append(truth_tree.query_ball_point(coordinate, radius))
        visited = np.zeros(len(truth_zyx), dtype=bool)
        if augment(prediction_index, visited):
            matched_count += 1
        matched_counts[prediction_index] = matched_count
    return matched_counts


def save_curve(
    run: Run,
    ranked: npt.NDArray[np.float64],
    matches: npt.NDArray[np.int64],
) -> None:
    path = OUTPUT_DIRECTORY / f"{run.key}_recall_curve.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ("rank_k", "overpick_factor", "matched", "recall", "precision", "score")
        )
        for index, (matched, score) in enumerate(zip(matches, ranked[:, 3]), start=1):
            writer.writerow(
                (
                    index,
                    index / PARTICLE_COUNT,
                    int(matched),
                    matched / PARTICLE_COUNT,
                    matched / index,
                    float(score),
                )
            )
    np.save(OUTPUT_DIRECTORY / f"{run.key}_ranked_nms_top10N.npy", ranked)


def plot_curves(results: dict[str, dict[str, object]]) -> None:
    """Create publication-ready recall and precision diagnostics."""
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.4), constrained_layout=True)
    factor_ticks = (1, 2, 3, 5, 10)

    for run in RUNS:
        result = results[run.key]
        matches = np.asarray(result["matches"])
        ranks = np.arange(1, len(matches) + 1)
        factors = ranks / PARTICLE_COUNT
        recall = matches / PARTICLE_COUNT
        precision = matches / ranks
        axes[0].plot(factors, recall, color=run.color, linewidth=2.2, label=run.label)
        axes[1].plot(factors, recall, color=run.color, linewidth=2.2, label=run.label)
        axes[2].plot(factors, precision, color=run.color, linewidth=2.2, label=run.label)
        for factor in factor_ticks:
            index = min(factor * PARTICLE_COUNT, len(matches)) - 1
            axes[0].scatter(factors[index], recall[index], color=run.color, s=24, zorder=3)

    axes[0].set(
        title="Full recall curve",
        xlabel="Retained picks / known particles ($K/N$)",
        ylabel="Recall",
        xlim=(0, MAXIMUM_FACTOR),
        ylim=(0, 1.02),
    )
    axes[0].set_xticks(factor_ticks)
    axes[1].set(
        title="Low-budget ranking (0–3× over-picking)",
        xlabel="Retained picks / known particles ($K/N$)",
        ylabel="Recall",
        xlim=(0, 3),
        ylim=(0, None),
    )
    axes[2].set(
        title="Precision cost of over-picking",
        xlabel="Retained picks / known particles ($K/N$)",
        ylabel="Precision",
        xlim=(0, MAXIMUM_FACTOR),
        ylim=(0, 1.02),
    )
    for axis in axes:
        axis.axvline(1, color="0.35", linestyle="--", linewidth=1, alpha=0.8)
        axis.legend(frameon=True, fontsize=9)
    figure.suptitle(
        "EMPIAR-10045 Tomogram 08: recall after global spherical NMS\n"
        f"N={PARTICLE_COUNT}, matching/NMS radius={MATCH_RADIUS_VOXELS:.2f} voxels",
        fontsize=14,
    )
    figure.savefig(OUTPUT_DIRECTORY / "full_tomogram_recall_k_curves.png", dpi=220)
    figure.savefig(OUTPUT_DIRECTORY / "full_tomogram_recall_k_curves.pdf")
    plt.close(figure)


def main() -> None:
    configure_logging()
    truth_xyz = np.loadtxt(TRUTH_PATH, dtype=np.float64, ndmin=2)
    truth_zyx = truth_xyz[:, ::-1]
    if len(truth_zyx) != PARTICLE_COUNT:
        raise ValueError(f"expected {PARTICLE_COUNT} annotations, found {len(truth_zyx)}")
    maximum_picks = MAXIMUM_FACTOR * PARTICLE_COUNT
    results: dict[str, dict[str, object]] = {}

    for run in RUNS:
        logging.info("RUN START | %s", run.label)
        candidates = np.load(run.candidates_path, mmap_mode="r")
        logging.info("Loaded %s candidate rows from %s", f"{len(candidates):,}", run.candidates_path)
        ranked, pool_count = extended_nms(
            candidates,
            radius=MATCH_RADIUS_VOXELS,
            max_picks=maximum_picks,
            initial_pool_size=run.initial_pool_size,
        )
        validate_saved_prefix(ranked, run.saved_particles_path)
        logging.info("Historical top-%d NMS prefix reproduced exactly", PARTICLE_COUNT)
        matches = incremental_recall_curve(ranked, truth_zyx, MATCH_RADIUS_VOXELS)
        save_curve(run, ranked, matches)
        checkpoints: dict[str, dict[str, float | int]] = {}
        for factor in (1, 2, 3, 5, 10):
            k = min(factor * PARTICLE_COUNT, len(ranked))
            matched = int(matches[k - 1])
            checkpoints[str(factor)] = {
                "k": k,
                "matched": matched,
                "recall": matched / PARTICLE_COUNT,
                "precision": matched / k,
            }
            logging.info(
                "%s | %dx: matched=%d/%d recall=%.4f precision=%.4f",
                run.key,
                factor,
                matched,
                PARTICLE_COUNT,
                matched / PARTICLE_COUNT,
                matched / k,
            )
        full_indices = np.flatnonzero(matches == PARTICLE_COUNT)
        results[run.key] = {
            "label": run.label,
            "candidate_count": len(candidates),
            "exact_score_pool_count": pool_count,
            "nms_pick_count": len(ranked),
            "matches": matches,
            "checkpoints": checkpoints,
            "first_full_recall_k": (
                None if not len(full_indices) else int(full_indices[0] + 1)
            ),
        }
        logging.info("RUN DONE | %s", run.label)

    plot_curves(results)
    serializable = {
        key: {name: value for name, value in result.items() if name != "matches"}
        for key, result in results.items()
    }
    with (OUTPUT_DIRECTORY / "summary.json").open("w") as stream:
        json.dump(serializable, stream, indent=2, sort_keys=True)
        stream.write("\n")
    logging.info("Saved recall analysis to %s", OUTPUT_DIRECTORY)


if __name__ == "__main__":
    main()
