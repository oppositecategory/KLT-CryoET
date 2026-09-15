"""Run an end-to-end one-GPU KLT experiment on a labeled EMPIAR crop."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import jax
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from kltpicker_3d.multi_gpu import (  # noqa: E402
    MultiGPUKLTParticleDetector3D,
    ranked_candidate_nms_3d,
)
from kltpicker_3d.streaming import (  # noqa: E402
    ArrayVolumeSource,
    MrcVolumeSource,
    SpatialRegion,
)


DEFAULT_DATASET_ROOT = Path(
    "/data/yoelsh/datasets/10045/pristine/data/ribosomes"
)
DEFAULT_TOMOGRAM = DEFAULT_DATASET_ROOT / "Tomograms/08/IS002_291013_008.mrc"
DEFAULT_TRUTH = (
    DEFAULT_DATASET_ROOT
    / "AnticipatedResults/Tomograms/08/IS002_291013_008.coords"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "results/empiar-10045-single-gpu-crop"
DEFAULT_START = (57, 1329, 531)
DEFAULT_SIDE = 372


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tomogram", type=Path, default=DEFAULT_TOMOGRAM)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--crop-start", type=int, nargs=3, default=DEFAULT_START)
    parser.add_argument("--crop-side", type=int, default=DEFAULT_SIDE)
    parser.add_argument("--voxel-size", type=float, default=2.2763967514038086)
    parser.add_argument("--particle-diameter", type=float, default=270.0)
    parser.add_argument("--valid-margin", type=int, default=86)
    parser.add_argument("--candidate-capacity", type=int, default=4096)
    return parser.parse_args()


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("empiar-single-gpu-crop")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "experiment.log", mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def recall_summary(
    predictions: np.ndarray,
    truth_zyx: np.ndarray,
    radius: float,
) -> dict[str, float | int | None]:
    if predictions.size == 0:
        matched_distances = np.empty(0)
    else:
        distances = cdist(predictions[:, :3], truth_zyx)
        within = distances <= radius
        predicted_indices, truth_indices = linear_sum_assignment(~within)
        accepted = within[predicted_indices, truth_indices]
        matched_distances = distances[
            predicted_indices[accepted], truth_indices[accepted]
        ]
    return {
        "requested_picks": int(len(predictions)),
        "truth_count": int(len(truth_zyx)),
        "matched": int(len(matched_distances)),
        "recall": float(len(matched_distances) / len(truth_zyx)),
        "mean_distance_voxels": (
            None if not len(matched_distances) else float(np.mean(matched_distances))
        ),
        "median_distance_voxels": (
            None if not len(matched_distances) else float(np.median(matched_distances))
        ),
    }


def timed(logger: logging.Logger, name: str, function):
    logger.info("STAGE START | %s", name)
    start = time.perf_counter()
    value = function()
    logger.info("STAGE DONE  | %s | %.2f s", name, time.perf_counter() - start)
    return value


def main() -> None:
    args = parse_args()
    logger = configure_logging(args.output_dir)
    devices = tuple(jax.devices())
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise RuntimeError(
            "This experiment requires exactly one visible GPU; set "
            "CUDA_VISIBLE_DEVICES to one device."
        )

    crop_start = np.asarray(args.crop_start, dtype=np.int64)
    crop_shape = np.full(3, args.crop_side, dtype=np.int64)
    crop_stop = crop_start + crop_shape
    truth_xyz = np.loadtxt(args.truth, dtype=np.float64, ndmin=2)
    truth_local = truth_xyz[:, ::-1] - crop_start
    safe = np.all(
        (truth_local >= args.valid_margin)
        & (truth_local < crop_shape - args.valid_margin),
        axis=1,
    )
    truth_local = truth_local[safe]
    if not len(truth_local):
        raise ValueError("selected crop has no safely interior annotations")

    with MrcVolumeSource(args.tomogram) as source:
        crop = np.asarray(
            source.read(
                SpatialRegion(
                    tuple(int(value) for value in crop_start),
                    tuple(int(value) for value in crop_stop),
                )
            ),
            dtype=np.float32,
        ).copy()
    source = ArrayVolumeSource(crop)
    patch_size = 93
    if args.crop_side % patch_size:
        raise ValueError("crop side must be divisible by the 93-voxel patch size")
    match_radius = 0.5 * args.particle_diameter / args.voxel_size

    detector = MultiGPUKLTParticleDetector3D(
        source,
        args.particle_diameter,
        1.0 / args.voxel_size,
        len(truth_local),
        whitening_support_radius=37,
        bandpass_low_fraction=0.05,
        bandpass_high_fraction=0.05,
        devices=devices,
        core_patch_shape=(2, 2, 2),
        patches_per_microbatch=1,
        candidate_capacity_per_subvolume=args.candidate_capacity,
        legendre_order=150,
        threshold=-np.inf,
        max_iter=500,
        max_order=4,
        template_energy_fraction=0.99,
        max_templates=1000,
        psd_patch_size=patch_size,
        fredholm_radius=0.4 * args.particle_diameter / args.voxel_size,
        template_side=97,
        nms_radius=match_radius,
        score_template_batch_size=1,
        score_memory_fraction=0.8,
        boundary_mode="constant",
    )
    logger.info("JAX %s | device=%s", jax.__version__, devices[0])
    logger.info(
        "Crop global start=%s stop=%s shape=%s | safe truth=%d",
        tuple(crop_start),
        tuple(crop_stop),
        crop.shape,
        len(truth_local),
    )
    logger.info(
        "Geometry: patch=93^3 grid=(4,4,4) | core=186^3 | "
        "whitening radius=37 | template=97^3 | valid margin=%d",
        args.valid_margin,
    )

    detector.bandpass_filter = timed(
        logger, "finite band-pass filter", detector.build_bandpass_filter
    )
    detector.initial_rpsds = timed(
        logger,
        "initial one-GPU RPSD extraction",
        lambda: detector.estimate_rpsds(
            detector.bandpass_filter,
            description="Single-GPU initial RPSDs",
        ),
    )
    detector.initial_model = timed(
        logger,
        "initial ALS",
        lambda: detector.fit_rpsds(detector.initial_rpsds),
    )
    detector.whitening_filter = timed(
        logger,
        "finite whitening filter",
        lambda: detector.build_whitening_filter(detector.initial_model.noise_psd),
    )
    detector.whitened_rpsds = timed(
        logger,
        "whitened one-GPU RPSD extraction",
        lambda: detector.estimate_rpsds(
            detector.whitening_filter,
            description="Single-GPU whitened RPSDs",
        ),
    )
    detector.whitened_model = timed(
        logger,
        "whitened ALS",
        lambda: detector.fit_rpsds(detector.whitened_rpsds),
    )
    detector.templates = timed(
        logger,
        "Fredholm solve and templates",
        lambda: detector.build_templates(detector.whitened_model.particle_psd),
    )
    timed(
        logger,
        "host block QR score model",
        lambda: detector.prepare_score_filters(
            detector.templates,
            detector.whitened_model.noise_variance,
            host_qr=True,
        ),
    )
    detector.candidates = timed(
        logger,
        "one-GPU streamed scoring",
        lambda: detector.score_candidates(
            detector.templates,
            detector.whitened_model.noise_variance,
            detector.whitening_filter,
        ),
    )

    valid = np.all(
        (detector.candidates[:, :3] >= args.valid_margin)
        & (detector.candidates[:, :3] < crop_shape - args.valid_margin),
        axis=1,
    )
    valid_candidates = detector.candidates[valid]
    maximum_picks = min(1000, len(valid_candidates))
    ranked = ranked_candidate_nms_3d(
        valid_candidates,
        radius=match_radius,
        max_picks=maximum_picks,
    )
    requested = len(truth_local)
    particles = ranked[:requested]
    evaluation = recall_summary(particles, truth_local, match_radius)
    recall_curve = {
        str(k): recall_summary(ranked[: min(k, len(ranked))], truth_local, match_radius)
        for k in (requested, 10, 20, 50, 100, 250, 500, 1000)
        if k >= requested
    }
    logger.info(
        "FINAL RECALL | matched=%d/%d | recall=%.4f | candidates=%d valid=%d",
        evaluation["matched"],
        evaluation["truth_count"],
        evaluation["recall"],
        len(detector.candidates),
        len(valid_candidates),
    )
    for k, summary in recall_curve.items():
        logger.info(
            "Recall@%s: %d/%d = %.4f",
            k,
            summary["matched"],
            summary["truth_count"],
            summary["recall"],
        )

    np.save(args.output_dir / "truth_local_zyx.npy", truth_local)
    np.save(args.output_dir / "particles_top_truth_count.npy", particles)
    np.save(args.output_dir / "ranked_nms_top1000.npy", ranked)
    np.save(args.output_dir / "initial_patch_rpsds.npy", detector.initial_rpsds.rpsds)
    np.save(args.output_dir / "whitened_patch_rpsds.npy", detector.whitened_rpsds.rpsds)
    np.savez(
        args.output_dir / "learned_model.npz",
        initial_particle_psd=detector.initial_model.particle_psd,
        initial_noise_psd=detector.initial_model.noise_psd,
        initial_noise_variance=detector.initial_model.noise_variance,
        particle_psd=detector.whitened_model.particle_psd,
        noise_psd=detector.whitened_model.noise_psd,
        noise_variance=detector.whitened_model.noise_variance,
        radial_eigenvalues=detector.model.radial_eigvals,
        template_eigenvalues=detector.model.eigvals,
    )
    summary = {
        "crop_start_global_zyx": crop_start.tolist(),
        "crop_stop_global_zyx": crop_stop.tolist(),
        "crop_shape": crop_shape.tolist(),
        "valid_margin": args.valid_margin,
        "match_radius_voxels": match_radius,
        "template_shape": list(detector.templates.shape),
        "evaluation": evaluation,
        "recall_curve": recall_curve,
    }
    with (args.output_dir / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
