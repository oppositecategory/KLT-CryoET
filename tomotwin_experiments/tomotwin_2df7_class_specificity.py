#!/usr/bin/env python3
"""Test whether the 2DF7-configured KLT detector actually favors 2DF7.

The KLT model is fitted exactly once on the selected tomogram ROI.  Its score
tensor is then reused for every molecular class.  For class k, N_k top local
maxima are retained, where N_k is that class's known GT count, and those picks
are matched one-to-one to that class's GT centers.

Ground-truth identities are used only for this post-hoc simulation evaluation;
they are never supplied to the detector or its ALS/KLT fitting stages.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from kltpicker_3d.tomogram import KLTParticleDetector3D
from kltpicker_3d.utils import ranked_local_maxima_nms_3d


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "tomotwin_2df7"

# Same optimized settings used by the end-to-end package experiment in
# tomotwin_2df7.ipynb.
PARTICLE_DIAMETER_VOXELS = 33.0
PSD_PATCH_SIZE = 25
FREDHOLM_RADIUS = 0.4 * PARTICLE_DIAMETER_VOXELS
TEMPLATE_SIDE = 2 * int(np.ceil(FREDHOLM_RADIUS)) + 1
NMS_RADIUS = 0.5 * PARTICLE_DIAMETER_VOXELS
MATCH_TOLERANCE = 0.5 * PARTICLE_DIAMETER_VOXELS
BANDPASS_LOW_FRACTION = 0.05
BANDPASS_HIGH_FRACTION = 0.05
LEGENDRE_ORDER = 40
MAX_ANGULAR_ORDER = 4
ALS_MAX_ITER = 500

ROI_START_ZYX = np.array([24, 16, 304], dtype=int)
ROI_SIZE = 128
TARGET_CLASS = "2df7"
PDB_ID_PATTERN = re.compile(r"^[0-9][A-Za-z0-9]{3}$")


@dataclass(frozen=True)
class ClassGroundTruth:
    label: str
    in_roi: np.ndarray
    fully_supported: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the 2DF7-configured KLT detector once, then measure recall "
            "independently for every molecular class at that class's known count."
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
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path for the fully-supported per-class summary.",
    )
    return parser.parse_args()


def load_roi_and_ground_truth(
    data_dir: Path,
) -> tuple[np.ndarray, list[ClassGroundTruth], list[str]]:
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
        roi_slices = tuple(
            slice(int(start), int(stop))
            for start, stop in zip(ROI_START_ZYX, roi_stop)
        )
        roi = np.asarray(mrc.data[roi_slices], dtype=np.float32).copy()

    columns = ["class", "x", "y", "z", "rot1", "rot2", "rot3"]
    positions = pd.read_csv(positions_path, header=None, names=columns)
    positions["class"] = positions["class"].astype(str)
    molecule_labels = sorted(
        label
        for label in positions["class"].unique()
        if PDB_ID_PATTERN.fullmatch(label)
    )
    excluded_labels = sorted(
        set(positions["class"].unique()) - set(molecule_labels)
    )
    if TARGET_CLASS not in molecule_labels:
        raise ValueError(f"Target class {TARGET_CLASS!r} is absent from GT file")

    # Use the same D/2 support rule as the notebook. This is slightly more
    # conservative than the 14-voxel valid-convolution template offset.
    support_margin = 0.5 * PARTICLE_DIAMETER_VOXELS
    class_ground_truth: list[ClassGroundTruth] = []
    for label in molecule_labels:
        xyz = positions.loc[
            positions["class"] == label, ["x", "y", "z"]
        ].to_numpy(dtype=float)
        zyx = xyz[:, ::-1]
        inside = np.all((zyx >= ROI_START_ZYX) & (zyx < roi_stop), axis=1)
        local = zyx[inside] - ROI_START_ZYX
        supported = np.all(
            (local >= support_margin) & (local < ROI_SIZE - support_margin),
            axis=1,
        )
        class_ground_truth.append(
            ClassGroundTruth(
                label=label,
                in_roi=local,
                fully_supported=local[supported],
            )
        )

    return roi, class_ground_truth, excluded_labels


def top_picks_from_fixed_score(
    score_volume: np.ndarray,
    count: int,
) -> np.ndarray:
    indices, values = ranked_local_maxima_nms_3d(
        score_volume,
        radius=NMS_RADIUS,
        max_picks=count,
    )
    centers = indices.astype(float) + TEMPLATE_SIDE // 2
    return np.column_stack((centers, values))


def match_at_known_count(
    picks: np.ndarray,
    ground_truth: np.ndarray,
) -> tuple[int, np.ndarray]:
    if len(ground_truth) == 0 or len(picks) == 0:
        return 0, np.empty(0, dtype=float)

    distances = cdist(picks[:, :3], ground_truth)
    invalid_cost = (
        max(len(picks), len(ground_truth)) + 1
    ) * (MATCH_TOLERANCE + 1)
    costs = np.where(
        distances <= MATCH_TOLERANCE,
        distances,
        invalid_cost,
    )
    pick_indices, gt_indices = linear_sum_assignment(costs)
    assigned = distances[pick_indices, gt_indices]
    valid = assigned <= MATCH_TOLERANCE
    return int(valid.sum()), assigned[valid]


def evaluate_classes(
    score_volume: np.ndarray,
    ground_truth: list[ClassGroundTruth],
    *,
    fully_supported: bool,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in ground_truth:
        centers = item.fully_supported if fully_supported else item.in_roi
        known_count = len(centers)
        picks = top_picks_from_fixed_score(score_volume, known_count)
        matched, matched_distances = match_at_known_count(picks, centers)
        rows.append(
            {
                "class": item.label,
                "known_count": known_count,
                "picks_returned": len(picks),
                "matched": matched,
                "recall": matched / known_count if known_count else np.nan,
                "mean_distance_vox": (
                    float(matched_distances.mean())
                    if matched_distances.size
                    else np.nan
                ),
                "max_distance_vox": (
                    float(matched_distances.max())
                    if matched_distances.size
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def print_table(title: str, table: pd.DataFrame) -> None:
    printable = table.copy()
    printable["recall"] = printable["recall"].map(
        lambda value: "n/a" if pd.isna(value) else f"{value:.3f}"
    )
    for column in ("mean_distance_vox", "max_distance_vox"):
        printable[column] = printable[column].map(
            lambda value: "n/a" if pd.isna(value) else f"{value:.2f}"
        )
    print(f"\n{title}")
    print("-" * len(title))
    print(printable.to_string(index=False))


def print_specificity_interpretation(summary: pd.DataFrame) -> None:
    usable = summary.dropna(subset=["recall"]).copy()
    target_row = usable.loc[usable["class"] == TARGET_CLASS]
    if target_row.empty:
        print(f"\nNo fully supported {TARGET_CLASS} centers were available.")
        return

    target_recall = float(target_row.iloc[0]["recall"])
    better = usable.loc[usable["recall"] > target_recall, "class"].tolist()
    tied = usable.loc[
        np.isclose(usable["recall"], target_recall), "class"
    ].tolist()
    rank = 1 + int((usable["recall"] > target_recall).sum())

    print("\nSpecificity readout")
    print("-------------------")
    print(
        f"{TARGET_CLASS} recall = {target_recall:.3f}; rank by recall = "
        f"{rank}/{len(usable)}."
    )
    if better:
        print("Classes with higher recall: " + ", ".join(better))
    if len(tied) > 1:
        print("Classes tied with 2df7: " + ", ".join(tied))
    if not better and tied == [TARGET_CLASS]:
        print(
            "In this ROI, the fixed score ranking favors 2df7 over every "
            "other molecular class by recall."
        )
    else:
        print(
            "The fixed score ranking is not uniquely 2df7-specific by this "
            "known-count recall test."
        )
    print(
        "Interpret with the per-class denominators shown above: small counts "
        "make recall coarse, and this is an ROI-specific simulation diagnostic."
    )


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()

    print("=" * 78)
    print("2DF7-CONFIGURED KLT SCORE: PER-MOLECULE SPECIFICITY TEST")
    print("=" * 78)
    print(
        "Goal: determine whether one detector run ranks 2df7 centers more "
        "strongly than centers of the other simulated molecule classes."
    )
    print(
        "Protocol: fit ALS/KLT once without labels; reuse its score tensor; "
        "for each class keep exactly N_class NMS peaks and compute one-to-one "
        "recall against that class only."
    )
    print(
        "Important: GT identities and counts are used only after scoring, "
        "for this controlled simulation evaluation."
    )

    roi, ground_truth, excluded_labels = load_roi_and_ground_truth(data_dir)
    count_table = pd.DataFrame(
        {
            "class": [item.label for item in ground_truth],
            "centers_in_roi": [len(item.in_roi) for item in ground_truth],
            "fully_supported": [
                len(item.fully_supported) for item in ground_truth
            ],
        }
    )

    print("\nDetector and ROI settings")
    print("-------------------------")
    print(f"data directory          : {data_dir}")
    print(
        "ROI start (z,y,x)       : "
        f"{tuple(int(value) for value in ROI_START_ZYX)}"
    )
    print(f"ROI shape               : {roi.shape}")
    print(f"particle diameter D     : {PARTICLE_DIAMETER_VOXELS:.1f} voxels")
    print(f"PSD patch size          : {PSD_PATCH_SIZE}")
    print(f"Fredholm radius         : {FREDHOLM_RADIUS:.1f} voxels (0.4D)")
    print(f"template side           : {TEMPLATE_SIDE}")
    print(f"NMS radius              : {NMS_RADIUS:.1f} voxels (0.5D)")
    print(f"matching tolerance      : {MATCH_TOLERANCE:.1f} voxels (0.5D)")
    print(f"bandpass removed        : 5% low / 5% high")
    print(f"ALS maximum iterations  : {ALS_MAX_ITER}")
    print(f"excluded non-molecules  : {', '.join(excluded_labels) or 'none'}")
    print("\nGround-truth counts (not supplied to detector fitting)")
    print("-------------------------------------------------------")
    print(count_table.to_string(index=False))

    # num_particles affects only the final pick count, not the learned model or
    # score tensor. One placeholder pick avoids an unnecessary large NMS pass;
    # all reported class-specific NMS passes are performed below.
    detector = KLTParticleDetector3D(
        roi,
        particle_diameter=PARTICLE_DIAMETER_VOXELS,
        mgscale=1.0,
        num_particles=1,
        legendre_order=LEGENDRE_ORDER,
        threshold=-np.inf,
        max_iter=ALS_MAX_ITER,
        max_order=MAX_ANGULAR_ORDER,
        psd_patch_size=PSD_PATCH_SIZE,
        fredholm_radius=FREDHOLM_RADIUS,
        template_side=TEMPLATE_SIDE,
        bandpass_low_fraction=BANDPASS_LOW_FRACTION,
        bandpass_high_fraction=BANDPASS_HIGH_FRACTION,
        nms_radius=NMS_RADIUS,
    )

    print("\nFitting ALS/KLT and constructing the score tensor once...")
    start = time.perf_counter()
    detector.process_tomogram()
    elapsed = time.perf_counter() - start
    score = np.asarray(detector.score_mat)
    if score.ndim != 3 or not np.all(np.isfinite(score)):
        raise RuntimeError("Detector produced a non-finite or invalid score tensor")
    expected_shape = tuple(np.asarray(roi.shape) - TEMPLATE_SIDE + 1)
    if score.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected score shape {score.shape}; expected {expected_shape}"
        )
    print(f"Detector finished in {elapsed:.2f} s")
    print(
        f"score tensor shape={score.shape}, min={score.min():.4g}, "
        f"max={score.max():.4g}, mean={score.mean():.4g}"
    )

    supported_summary = evaluate_classes(
        score, ground_truth, fully_supported=True
    )
    all_roi_summary = evaluate_classes(
        score, ground_truth, fully_supported=False
    )
    print_table(
        "Primary comparison: fully supported centers (fair specificity test)",
        supported_summary,
    )
    print_table(
        "Boundary stress test: all centers whose GT center lies in the ROI",
        all_roi_summary,
    )
    print(
        "\nThe boundary table is intentionally secondary: a molecule centered "
        "near an ROI edge may be partly absent or outside the valid score-center "
        "domain, reducing recall for geometric reasons."
    )
    print_specificity_interpretation(supported_summary)

    if args.output_csv is not None:
        output_path = args.output_csv.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        supported_summary.to_csv(
            output_path,
            index=False,
            quoting=csv.QUOTE_MINIMAL,
        )
        print(f"\nSaved fully-supported summary to: {output_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
