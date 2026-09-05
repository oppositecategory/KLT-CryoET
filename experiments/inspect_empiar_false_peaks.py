"""Inspect high-scoring EMPIAR-10045 detections in orthogonal views.

The script compares distinct high-scoring false detections with depth-matched
annotated ribosomes and background locations.  It memory-maps the tomogram, so
only the small diagnostic cubes are materialized.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mrcfile
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path(
    "/data/yoelsh/datasets/10045/pristine/data/ribosomes"
)
DEFAULT_TOMOGRAM = (
    DEFAULT_DATASET_ROOT / "Tomograms/08/IS002_291013_008.mrc"
)
DEFAULT_TRUTH = (
    DEFAULT_DATASET_ROOT
    / "AnticipatedResults/Tomograms/08/IS002_291013_008.coords"
)
DEFAULT_CANDIDATES = (
    REPOSITORY_ROOT
    / "results/empiar-10045-bandpass-block-qr/07_candidates_top4096.npy"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT / "results/empiar-10045-bandpass-block-qr/diagnostics"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tomogram", type=Path, default=DEFAULT_TOMOGRAM)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=9)
    parser.add_argument("--radius", type=int, default=60)
    parser.add_argument("--slab-half-width", type=int, default=2)
    parser.add_argument("--match-radius", type=float, default=59.304)
    parser.add_argument("--seed", type=int, default=10045)
    return parser.parse_args()


def distances(points: np.ndarray, point: np.ndarray) -> np.ndarray:
    return np.linalg.norm(points - point[None, :], axis=1)


def select_false_peaks(
    candidates: np.ndarray,
    truth_zyx: np.ndarray,
    shape: tuple[int, int, int],
    count: int,
    crop_radius: int,
    match_radius: float,
) -> np.ndarray:
    # A modest top pool is sufficient and avoids sorting all ~5 million rows.
    pool_size = min(len(candidates), max(50_000, count * 5_000))
    indices = np.argpartition(candidates[:, 3], -pool_size)[-pool_size:]
    pool = np.asarray(candidates[indices])
    pool = pool[np.argsort(pool[:, 3])[::-1]]

    selected: list[np.ndarray] = []
    upper = np.asarray(shape) - crop_radius
    for row in pool:
        center = row[:3]
        if np.any(center < crop_radius) or np.any(center >= upper):
            continue
        if distances(truth_zyx, center).min() <= match_radius:
            continue
        if selected and distances(np.asarray(selected)[:, :3], center).min() <= match_radius:
            continue
        selected.append(row.copy())
        if len(selected) == count:
            return np.asarray(selected)
    raise RuntimeError(f"Found only {len(selected)} suitable false peaks")


def select_depth_matched_truth(
    false_peaks: np.ndarray,
    truth_zyx: np.ndarray,
    shape: tuple[int, int, int],
    crop_radius: int,
) -> np.ndarray:
    upper = np.asarray(shape) - crop_radius
    valid = np.all((truth_zyx >= crop_radius) & (truth_zyx < upper), axis=1)
    available = list(np.flatnonzero(valid))
    selected = []
    for false_peak in false_peaks:
        index = min(available, key=lambda i: abs(truth_zyx[i, 0] - false_peak[0]))
        selected.append(truth_zyx[index])
        available.remove(index)
    return np.asarray(selected)


def select_background(
    reference: np.ndarray,
    truth_zyx: np.ndarray,
    false_peaks: np.ndarray,
    shape: tuple[int, int, int],
    crop_radius: int,
    exclusion_radius: float,
    rng: np.random.Generator,
) -> np.ndarray:
    selected = []
    for point in reference:
        for _ in range(100_000):
            center = np.array(
                [
                    int(point[0]),
                    rng.integers(crop_radius, shape[1] - crop_radius),
                    rng.integers(crop_radius, shape[2] - crop_radius),
                ],
                dtype=float,
            )
            if distances(truth_zyx, center).min() <= exclusion_radius:
                continue
            if distances(false_peaks[:, :3], center).min() <= exclusion_radius:
                continue
            if selected and distances(np.asarray(selected), center).min() <= exclusion_radius:
                continue
            selected.append(center)
            break
        else:
            raise RuntimeError("Could not sample an isolated background location")
    return np.asarray(selected)


def extract_views(
    volume: np.ndarray,
    center: np.ndarray,
    radius: int,
    slab_half_width: int,
) -> tuple[np.ndarray, list[np.ndarray], dict[str, float]]:
    z, y, x = np.rint(center).astype(int)
    cube = np.asarray(
        volume[
            z - radius : z + radius + 1,
            y - radius : y + radius + 1,
            x - radius : x + radius + 1,
        ],
        dtype=np.float32,
    )
    middle = radius
    band = slice(middle - slab_half_width, middle + slab_half_width + 1)
    views = [
        cube[band, :, :].mean(axis=0),
        cube[:, band, :].mean(axis=1),
        cube[:, :, band].mean(axis=2),
    ]
    median = float(np.median(cube))
    std = float(np.std(cube))
    metrics = {
        "mean": float(np.mean(cube)),
        "std": std,
        "minimum": float(np.min(cube)),
        "maximum": float(np.max(cube)),
        "p01": float(np.percentile(cube, 0.1)),
        "p999": float(np.percentile(cube, 99.9)),
        "extreme_z": float(
            max(abs(np.min(cube) - median), abs(np.max(cube) - median))
            / max(std, np.finfo(np.float32).eps)
        ),
    }
    return cube, views, metrics


def make_montage(
    volume: np.ndarray,
    points: np.ndarray,
    labels: list[str],
    title: str,
    output: Path,
    radius: int,
    slab_half_width: int,
) -> list[dict[str, float | str]]:
    figure, axes = plt.subplots(
        len(points), 3, figsize=(10.5, 2.9 * len(points)), squeeze=False
    )
    records: list[dict[str, float | str]] = []
    view_names = ("XY", "XZ", "YZ")
    for row, (point, label) in enumerate(zip(points, labels)):
        cube, views, metrics = extract_views(
            volume, point[:3], radius, slab_half_width
        )
        low, high = np.percentile(cube, (0.5, 99.5))
        for column, (axis, view, name) in enumerate(
            zip(axes[row], views, view_names)
        ):
            axis.imshow(view, cmap="gray", vmin=low, vmax=high, origin="lower")
            axis.axhline(radius, color="tab:red", alpha=0.45, linewidth=0.6)
            axis.axvline(radius, color="tab:red", alpha=0.45, linewidth=0.6)
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(name)
            if column == 0:
                axis.set_ylabel(label, fontsize=8)
        records.append(
            {
                "group": title,
                "label": label,
                "z": float(point[0]),
                "y": float(point[1]),
                "x": float(point[2]),
                **metrics,
            }
        )
    figure.suptitle(title, fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return records


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    truth_xyz = np.loadtxt(args.truth, dtype=float, ndmin=2)
    truth_zyx = truth_xyz[:, ::-1]
    candidates = np.load(args.candidates, mmap_mode="r")
    rng = np.random.default_rng(args.seed)

    with mrcfile.mmap(args.tomogram, permissive=True, mode="r") as mrc:
        volume = mrc.data
        shape = tuple(int(value) for value in volume.shape)
        false_peaks = select_false_peaks(
            candidates,
            truth_zyx,
            shape,
            args.count,
            args.radius,
            args.match_radius,
        )
        matched_truth = select_depth_matched_truth(
            false_peaks, truth_zyx, shape, args.radius
        )
        background = select_background(
            false_peaks,
            truth_zyx,
            false_peaks,
            shape,
            args.radius,
            2.0 * args.radius,
            rng,
        )

        records = make_montage(
            volume,
            false_peaks,
            [
                f"{i + 1}: ({int(p[0])},{int(p[1])},{int(p[2])})  S={p[3]:.0f}"
                for i, p in enumerate(false_peaks)
            ],
            "Highest-scoring distinct false clusters",
            args.output_dir / "false_peaks_orthogonal.png",
            args.radius,
            args.slab_half_width,
        )
        records += make_montage(
            volume,
            matched_truth,
            [f"{i + 1}: ({int(p[0])},{int(p[1])},{int(p[2])})" for i, p in enumerate(matched_truth)],
            "Depth-matched annotated ribosomes",
            args.output_dir / "truth_orthogonal.png",
            args.radius,
            args.slab_half_width,
        )
        records += make_montage(
            volume,
            background,
            [f"{i + 1}: ({int(p[0])},{int(p[1])},{int(p[2])})" for i, p in enumerate(background)],
            "Depth-matched background",
            args.output_dir / "background_orthogonal.png",
            args.radius,
            args.slab_half_width,
        )

    metrics_path = args.output_dir / "orthogonal_metrics.csv"
    with metrics_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    np.save(args.output_dir / "false_peaks_zyx_score.npy", false_peaks)
    np.save(args.output_dir / "truth_depth_matched_zyx.npy", matched_truth)
    np.save(args.output_dir / "background_depth_matched_zyx.npy", background)
    print(f"Wrote diagnostic montages and metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
