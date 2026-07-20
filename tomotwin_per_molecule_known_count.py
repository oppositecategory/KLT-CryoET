#!/usr/bin/env python3
"""Run one class-scale KLT detector per TomoTwin molecule.

This is a controlled simulation experiment with known molecule counts:

1. For molecule k, configure the complete ALS/KLT detector using TomoTwin's
   class-specific evaluation box size D_k.
2. Fit and score the same tomogram ROI from scratch at that scale.
3. Retain exactly N_k picks, where N_k is the number of GT centers of molecule
   k inside the ROI.
4. Match the N_k picks one-to-one to molecule k and report recall.

GT identities and counts are used only to select the scale and evaluate the
known-count experiment. They are not used by ALS or KLT spectrum estimation.

The seven sizes below are from ``data/boxsizes.json`` in the official TomoTwin
Demonstration Dataset (Zenodo record 7225386). The source file calls them box
sizes; this experiment uses each value as the corresponding KLT diameter in
voxels. All reference subvolumes themselves have a common 37^3 array shape.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mrcfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from kltpicker_3d.tomogram import KLTParticleDetector3D


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "tomotwin_2df7"

# TomoTwin demo data/boxsizes.json, matched case-insensitively to the GT labels.
TOMOTWIN_EVALUATION_BOX_VOXELS = {
    "1avo": 18,
    "1e9r": 21,
    "1fpy": 18,
    "1fzg": 28,
    "1jz8": 20,
    "1oao": 20,
    "2df7": 33,
}

# Molecular masses shown for the seven generalization-tomogram proteins in
# Fig. 5 of the TomoTwin paper.
MOLECULAR_MASS_KDA = {
    "1avo": 149,
    "1e9r": 276,
    "1fpy": 632,
    "1fzg": 142,
    "1jz8": 512,
    "1oao": 315,
    "2df7": 896,
}

ROI_START_ZYX = np.array([24, 16, 304], dtype=int)
ROI_SIZE = 128

LEGENDRE_ORDER = 40
MAX_ANGULAR_ORDER = 4
ALS_MAX_ITER = 500
BANDPASS_LOW_FRACTION = 0.05
BANDPASS_HIGH_FRACTION = 0.05


@dataclass(frozen=True)
class ClassExperiment:
    label: str
    diameter: int
    gt_in_roi: np.ndarray
    gt_supported: np.ndarray

    @property
    def psd_patch_size(self) -> int:
        value = int(np.floor(0.8 * self.diameter))
        return value if value % 2 else value - 1

    @property
    def fredholm_radius(self) -> float:
        return 0.4 * self.diameter

    @property
    def template_side(self) -> int:
        return 2 * int(np.ceil(self.fredholm_radius)) + 1

    @property
    def nms_radius(self) -> float:
        return 0.5 * self.diameter

    @property
    def match_tolerance(self) -> float:
        return 0.5 * self.diameter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun ALS/KLT separately for every TomoTwin molecule using its "
            "evaluation box size, then report recall at the known class count."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing tiltseries_rec.mrc and particle_positions.txt.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=str.lower,
        default=list(TOMOTWIN_EVALUATION_BOX_VOXELS),
        help="Molecule classes to run; useful for a shorter diagnostic.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path at which to save the per-class summary.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=PROJECT_ROOT / "tomotwin_per_molecule_cross_matches.png",
        help="Path for the cross-molecule count heatmap and box-size plot.",
    )
    args = parser.parse_args()
    unknown = sorted(
        set(args.classes) - set(TOMOTWIN_EVALUATION_BOX_VOXELS)
    )
    if unknown:
        parser.error("unknown molecule class(es): " + ", ".join(unknown))
    # Preserve the official mapping order and remove duplicate CLI entries.
    requested = set(args.classes)
    args.classes = [
        label
        for label in TOMOTWIN_EVALUATION_BOX_VOXELS
        if label in requested
    ]
    return args


def load_experiments(
    data_dir: Path,
    selected_classes: list[str],
) -> tuple[np.ndarray, list[ClassExperiment]]:
    tomogram_path = data_dir / "tiltseries_rec.mrc"
    positions_path = data_dir / "particle_positions.txt"
    if not tomogram_path.is_file():
        raise FileNotFoundError(f"Missing tomogram: {tomogram_path}")
    if not positions_path.is_file():
        raise FileNotFoundError(f"Missing positions: {positions_path}")

    roi_stop = ROI_START_ZYX + ROI_SIZE
    with mrcfile.open(tomogram_path, permissive=True) as mrc:
        full_shape = np.asarray(mrc.data.shape)
        if np.any(ROI_START_ZYX < 0) or np.any(roi_stop > full_shape):
            raise ValueError(
                f"ROI {tuple(ROI_START_ZYX)}:{tuple(roi_stop)} exceeds "
                f"tomogram shape {tuple(full_shape)}"
            )
        slices = tuple(
            slice(int(start), int(stop))
            for start, stop in zip(ROI_START_ZYX, roi_stop)
        )
        roi = np.asarray(mrc.data[slices], dtype=np.float32).copy()

    columns = ["class", "x", "y", "z", "rot1", "rot2", "rot3"]
    positions = pd.read_csv(positions_path, header=None, names=columns)
    positions["class"] = positions["class"].astype(str).str.lower()

    experiments: list[ClassExperiment] = []
    for label in selected_classes:
        diameter = TOMOTWIN_EVALUATION_BOX_VOXELS[label]
        xyz = positions.loc[
            positions["class"] == label, ["x", "y", "z"]
        ].to_numpy(dtype=float)
        if len(xyz) == 0:
            raise ValueError(f"No GT entries found for class {label}")
        zyx = xyz[:, ::-1]
        inside = np.all((zyx >= ROI_START_ZYX) & (zyx < roi_stop), axis=1)
        local = zyx[inside] - ROI_START_ZYX

        # Same conservative support rule as the 2DF7 notebook, now scaled by
        # each molecule's own TomoTwin evaluation box.
        margin = 0.5 * diameter
        supported = np.all(
            (local >= margin) & (local < ROI_SIZE - margin),
            axis=1,
        )
        experiments.append(
            ClassExperiment(
                label=label,
                diameter=diameter,
                gt_in_roi=local,
                gt_supported=local[supported],
            )
        )
    return roi, experiments


def match_picks(
    picks: np.ndarray,
    ground_truth: np.ndarray,
    tolerance: float,
) -> tuple[int, np.ndarray]:
    """Maximize valid one-to-one matches, then minimize their distances."""
    if len(picks) == 0 or len(ground_truth) == 0:
        return 0, np.empty(0, dtype=float)
    distances = cdist(picks[:, :3], ground_truth)
    invalid_cost = (
        max(len(picks), len(ground_truth)) + 1
    ) * (tolerance + 1)
    costs = np.where(distances <= tolerance, distances, invalid_cost)
    pick_indices, gt_indices = linear_sum_assignment(costs)
    assigned = distances[pick_indices, gt_indices]
    valid = assigned <= tolerance
    return int(valid.sum()), assigned[valid]


def assign_picks_to_all_classes(
    picks: np.ndarray,
    experiments: list[ClassExperiment],
    tolerance: float,
) -> tuple[dict[str, int], int]:
    """Assign every pick to at most one GT instance across all molecule classes."""
    counts = {item.label: 0 for item in experiments}
    if len(picks) == 0:
        return counts, 0

    all_centers = np.concatenate(
        [item.gt_in_roi for item in experiments],
        axis=0,
    )
    all_labels = np.concatenate(
        [
            np.repeat(item.label, len(item.gt_in_roi))
            for item in experiments
        ]
    )
    distances = cdist(picks[:, :3], all_centers)
    invalid_cost = (
        max(len(picks), len(all_centers)) + 1
    ) * (tolerance + 1)
    costs = np.where(distances <= tolerance, distances, invalid_cost)
    pick_indices, gt_indices = linear_sum_assignment(costs)
    assigned = distances[pick_indices, gt_indices]
    valid = assigned <= tolerance
    for label in all_labels[gt_indices[valid]]:
        counts[str(label)] += 1
    return counts, int(len(picks) - valid.sum())


def run_experiment(
    roi: np.ndarray,
    experiment: ClassExperiment,
    all_experiments: list[ClassExperiment],
) -> dict[str, object]:
    known_count = len(experiment.gt_in_roi)
    if known_count == 0:
        empty_row: dict[str, object] = {
            "class": experiment.label,
            "mass_kda": MOLECULAR_MASS_KDA[experiment.label],
            "diameter_vox": experiment.diameter,
            "psd_patch_size": experiment.psd_patch_size,
            "fredholm_radius_vox": experiment.fredholm_radius,
            "template_side": experiment.template_side,
            "centers_in_roi": len(experiment.gt_in_roi),
            "known_count": 0,
            "fully_supported_count": len(experiment.gt_supported),
            "picks_returned": 0,
            "matched": 0,
            "recall": np.nan,
            "mean_distance_vox": np.nan,
            "max_distance_vox": np.nan,
            "elapsed_seconds": 0.0,
            "score_max": np.nan,
        }
        empty_row.update(
            {f"found_{item.label}": 0 for item in all_experiments}
        )
        empty_row["found_unmatched"] = 0
        return empty_row

    detector = KLTParticleDetector3D(
        roi,
        particle_diameter=experiment.diameter,
        mgscale=1.0,
        num_particles=known_count,
        legendre_order=LEGENDRE_ORDER,
        threshold=-np.inf,
        max_iter=ALS_MAX_ITER,
        max_order=MAX_ANGULAR_ORDER,
        psd_patch_size=experiment.psd_patch_size,
        fredholm_radius=experiment.fredholm_radius,
        template_side=experiment.template_side,
        bandpass_low_fraction=BANDPASS_LOW_FRACTION,
        bandpass_high_fraction=BANDPASS_HIGH_FRACTION,
        nms_radius=experiment.nms_radius,
    )

    start = time.perf_counter()
    num_picks, picks = detector.process_tomogram()
    elapsed = time.perf_counter() - start
    score = np.asarray(detector.score_mat)
    if score.ndim != 3 or not np.all(np.isfinite(score)):
        raise RuntimeError(
            f"{experiment.label}: detector produced an invalid score tensor"
        )
    if num_picks != known_count:
        raise RuntimeError(
            f"{experiment.label}: requested {known_count} picks, got {num_picks}"
        )

    matched, distances = match_picks(
        picks,
        experiment.gt_in_roi,
        experiment.match_tolerance,
    )
    found_counts, unmatched = assign_picks_to_all_classes(
        picks,
        all_experiments,
        experiment.match_tolerance,
    )
    row: dict[str, object] = {
        "class": experiment.label,
        "mass_kda": MOLECULAR_MASS_KDA[experiment.label],
        "diameter_vox": experiment.diameter,
        "psd_patch_size": experiment.psd_patch_size,
        "fredholm_radius_vox": experiment.fredholm_radius,
        "template_side": experiment.template_side,
        "centers_in_roi": len(experiment.gt_in_roi),
        "known_count": known_count,
        "fully_supported_count": len(experiment.gt_supported),
        "picks_returned": num_picks,
        "matched": matched,
        "recall": matched / known_count,
        "mean_distance_vox": (
            float(distances.mean()) if distances.size else np.nan
        ),
        "max_distance_vox": (
            float(distances.max()) if distances.size else np.nan
        ),
        "elapsed_seconds": elapsed,
        "score_max": float(score.max()),
    }
    row.update(
        {
            f"found_{label}": count
            for label, count in found_counts.items()
        }
    )
    row["found_unmatched"] = unmatched
    return row


def format_result_line(row: dict[str, object]) -> str:
    recall = row["recall"]
    recall_text = "n/a" if pd.isna(recall) else f"{float(recall):.3f}"
    return (
        f"{row['class']}: D={row['diameter_vox']} vox, "
        f"PSD patch={row['psd_patch_size']}, "
        f"GT={row['known_count']}, "
        f"matched={row['matched']}/{row['known_count']}, "
        f"recall={recall_text}, time={float(row['elapsed_seconds']):.2f} s"
    )


def print_summary(summary: pd.DataFrame) -> None:
    columns = [
        "class",
        "mass_kda",
        "diameter_vox",
        "psd_patch_size",
        "fredholm_radius_vox",
        "template_side",
        "centers_in_roi",
        "known_count",
        "fully_supported_count",
        "picks_returned",
        "matched",
        "recall",
        "mean_distance_vox",
        "max_distance_vox",
        "elapsed_seconds",
    ]
    printable = summary[columns].copy()
    for column in (
        "recall",
        "mean_distance_vox",
        "max_distance_vox",
        "elapsed_seconds",
    ):
        printable[column] = printable[column].map(
            lambda value: "n/a" if pd.isna(value) else f"{value:.3f}"
        )
    printable["fredholm_radius_vox"] = printable[
        "fredholm_radius_vox"
    ].map(lambda value: f"{value:.1f}")

    print("\n" + "=" * 110)
    print("FINAL PER-MOLECULE KNOWN-COUNT RESULTS")
    print("=" * 110)
    print(printable.to_string(index=False))

    usable = summary.dropna(subset=["recall"])
    total_gt = int(usable["known_count"].sum())
    total_matched = int(usable["matched"].sum())
    macro_recall = float(usable["recall"].mean())
    micro_recall = total_matched / total_gt if total_gt else np.nan
    print(
        f"\nMacro recall across classes: {macro_recall:.3f}\n"
        f"Micro recall across ROI instances: "
        f"{total_matched}/{total_gt} = {micro_recall:.3f}"
    )
    print(
        "This is a known-count localization test at the supplied class scale. "
        "Its outcome does not address how to infer class identity, diameter, "
        "or count without GT."
    )


def cross_match_table(
    summary: pd.DataFrame,
    class_labels: list[str],
) -> pd.DataFrame:
    columns = [f"found_{label}" for label in class_labels]
    table = summary[["class", "diameter_vox", *columns, "found_unmatched"]].copy()
    table = table.rename(
        columns={
            **{f"found_{label}": label for label in class_labels},
            "found_unmatched": "unmatched",
        }
    )
    return table


def plot_cross_matches(
    summary: pd.DataFrame,
    class_labels: list[str],
    output_path: Path,
) -> None:
    matrix_columns = [f"found_{label}" for label in class_labels]
    matrix = summary[matrix_columns + ["found_unmatched"]].to_numpy(dtype=int)
    column_labels = [*class_labels, "unmatched"]
    row_labels = [
        f"{row['class']}  (D={int(row['diameter_vox'])})"
        for _, row in summary.iterrows()
    ]

    fig, (heat_ax, box_ax) = plt.subplots(
        1,
        2,
        figsize=(13.5, 6.2),
        gridspec_kw={"width_ratios": [3.5, 1.25]},
        constrained_layout=True,
    )
    image = heat_ax.imshow(matrix, cmap="magma", aspect="auto", vmin=0)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            heat_ax.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > matrix.max() * 0.35 else "black",
                fontsize=10,
            )
    for index in range(min(len(class_labels), len(summary))):
        heat_ax.add_patch(
            plt.Rectangle(
                (index - 0.5, index - 0.5),
                1,
                1,
                fill=False,
                edgecolor="cyan",
                linewidth=1.8,
            )
        )
    heat_ax.set_xticks(np.arange(len(column_labels)), column_labels, rotation=35)
    heat_ax.set_yticks(np.arange(len(row_labels)), row_labels)
    heat_ax.set_xlabel("GT identity assigned to each returned pick")
    heat_ax.set_ylabel("Detector run and TomoTwin evaluation box")
    heat_ax.set_title("Top-known-count pick composition")
    fig.colorbar(image, ax=heat_ax, label="one-to-one matched picks")

    y = np.arange(len(summary))
    diameters = summary["diameter_vox"].to_numpy(dtype=float)
    masses = summary["mass_kda"].to_numpy(dtype=int)
    box_ax.barh(y, diameters, color="steelblue", alpha=0.85)
    for index, (diameter, mass) in enumerate(zip(diameters, masses)):
        box_ax.text(
            diameter + 0.5,
            index,
            f"{int(diameter)} vox\n{mass} kDa",
            va="center",
            fontsize=9,
        )
    box_ax.set_yticks(y, summary["class"])
    box_ax.invert_yaxis()
    box_ax.set_xlim(0, max(diameters) + 14)
    box_ax.set_xlabel("evaluation box D (voxels)")
    box_ax.set_title("Scale and molecular mass")
    box_ax.grid(axis="x", alpha=0.25)

    fig.suptitle(
        "Independent ALS/KLT fits: which molecule identities occupy each top-N list?"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    roi, experiments = load_experiments(data_dir, args.classes)

    print("=" * 78)
    print("TOMOTWIN PER-MOLECULE SCALE: KNOWN-COUNT ALS/KLT EXPERIMENT")
    print("=" * 78)
    print(
        "Goal: first test whether ALS/KLT can localize each molecule when its "
        "TomoTwin evaluation box and its number of instances in the ROI are known."
    )
    print(
        "Every row is a fresh end-to-end fit on the same ROI. No score tensor "
        "is reused between molecule classes."
    )
    print(
        "GT is used only to select D_k, set N_k, and evaluate the resulting "
        "top-N_k picks."
    )
    print(
        "No PDB identity, reference density, or molecule-specific template is "
        "provided to ALS/KLT."
    )
    print(f"\ndata directory       : {data_dir}")
    print(
        "ROI start (z,y,x)  : "
        f"{tuple(int(value) for value in ROI_START_ZYX)}"
    )
    print(f"ROI shape            : {roi.shape}")
    print("bandpass             : remove 5% low / 5% high")
    print(f"ALS max iterations   : {ALS_MAX_ITER}")

    setup_rows = []
    for item in experiments:
        setup_rows.append(
            {
                "class": item.label,
                "mass_kDa": MOLECULAR_MASS_KDA[item.label],
                "D_vox": item.diameter,
                "PSD_patch": item.psd_patch_size,
                "Fredholm_radius": item.fredholm_radius,
                "template_side": item.template_side,
                "N_in_ROI": len(item.gt_in_roi),
                "N_supported": len(item.gt_supported),
                "NMS_radius": item.nms_radius,
                "match_tolerance": item.match_tolerance,
            }
        )
    print("\nPlanned independent runs")
    print("------------------------")
    print(pd.DataFrame(setup_rows).to_string(index=False))

    results: list[dict[str, object]] = []
    total_start = time.perf_counter()
    for index, experiment in enumerate(experiments, start=1):
        print(
            f"\n[{index}/{len(experiments)}] Running {experiment.label} "
            f"from scratch at D={experiment.diameter} voxels...",
            flush=True,
        )
        result = run_experiment(roi, experiment, experiments)
        results.append(result)
        print(format_result_line(result), flush=True)

    summary = pd.DataFrame(results)
    print_summary(summary)
    labels = [item.label for item in experiments]
    composition = cross_match_table(summary, labels)
    print("\nCross-molecule identity of the returned picks")
    print("---------------------------------------------")
    print(
        "Rows are independent detector runs; columns are the GT molecule "
        "identities assigned one-to-one within that run's D/2 tolerance."
    )
    print(composition.to_string(index=False))

    plot_path = args.output_plot.expanduser().resolve()
    plot_cross_matches(summary, labels, plot_path)
    print(f"\nSaved cross-match plot to: {plot_path}")
    print(f"Total execution time: {time.perf_counter() - total_start:.2f} s")

    if args.output_csv is not None:
        output_path = args.output_csv.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"Saved summary to: {output_path}")
        composition_path = output_path.with_name(
            output_path.stem + "_cross_matches" + output_path.suffix
        )
        composition.to_csv(
            composition_path,
            index=False,
            quoting=csv.QUOTE_MINIMAL,
        )
        print(f"Saved cross-match table to: {composition_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
