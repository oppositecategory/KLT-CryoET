"""Run the complete multi-GPU KLT experiment on EMPIAR-10045 tomogram 08."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pickle
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import jax
import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from kltpicker_3d.fredholm_solver import INVERSE_FOURIER_NORMALIZATION_3D
from kltpicker_3d.multi_gpu import MultiGPUKLTParticleDetector3D
from kltpicker_3d.streaming import MrcVolumeSource

# DATASET_ROOT = Path(
#     "/luke_leia_data/yoelsh/datasets/10045/pristine/data/ribosomes"
# )
# DEFAULT_INPUT = DATASET_ROOT / "Tomograms/08/IS002_291013_008.mrc"
# DEFAULT_GROUND_TRUTH = (
#     DATASET_ROOT
#     / "AnticipatedResults/Tomograms/08/IS002_291013_008.coords"
# )
# DEFAULT_RESULTS_DIR = REPOSITORY_ROOT / "results/empiar-10045-bandpass-block-qr"


DATASET_ROOT = Path(
    "/data/yoelsh/datasets/10045/pristine/data/ribosomes"
)
DEFAULT_INPUT = DATASET_ROOT / "Tomograms/08/IS002_291013_008.mrc"
DEFAULT_GROUND_TRUTH = (
    DATASET_ROOT
    / "AnticipatedResults/Tomograms/08/IS002_291013_008.coords"
)
DEFAULT_RESULTS_DIR = REPOSITORY_ROOT / "results/empiar-10045-bandpass-block-qr"

LOGGER = logging.getLogger("empiar-10045")
T = TypeVar("T")
_SCORE_MODEL_METHOD = "block_qr_nonnegative_m_v3_fourier_normalized"


def parse_args() -> argparse.Namespace:
    """Parse experiment, checkpointing, and evaluation settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Default: <results-dir>/empiar-10045.log",
    )
    parser.add_argument(
        "--initial-rpsds",
        type=Path,
        help=(
            "Existing band-pass first-pass checkpoint; ignored with "
            "--recompute-initial. Raw legacy RPSDs are not compatible."
        ),
    )
    parser.add_argument("--recompute-initial", action="store_true")
    parser.add_argument("--particle-diameter", type=float, default=270.0)
    parser.add_argument(
        "--voxel-size",
        type=float,
        metavar="ANGSTROM",
        help="Override the isotropic MRC voxel size.",
    )
    parser.add_argument(
        "--whitening-support-radius",
        type=int,
        default=37,
        metavar="VOXELS",
    )
    parser.add_argument("--bandpass-low-fraction", type=float, default=0.05)
    parser.add_argument("--bandpass-high-fraction", type=float, default=0.05)
    parser.add_argument(
        "--match-radius-angstrom",
        type=float,
        help="Recall radius; default: half the particle diameter.",
    )
    parser.add_argument(
        "--core-patch-shape",
        type=int,
        nargs=3,
        metavar=("Z", "Y", "X"),
        help="Patches per GPU core; default: plan for the largest pipeline halo.",
    )
    parser.add_argument("--memory-fraction", type=float, default=0.3)
    parser.add_argument("--resident-volume-copies", type=int, default=8)
    parser.add_argument("--patches-per-microbatch", type=int, default=1)
    parser.add_argument(
        "--candidate-capacity-per-subvolume",
        type=int,
        default=4_096,
        help=(
            "Number of highest 3x3x3 local maxima retained per subvolume "
            "before global ranking and NMS."
        ),
    )
    parser.add_argument("--legendre-order", type=int, default=150)
    parser.add_argument(
        "--max-order",
        type=int,
        default=4,
        help="Number of angular orders; default retains ell=0,1,2,3.",
    )
    parser.add_argument("--template-energy-fraction", type=float, default=0.99)
    parser.add_argument("--max-templates", type=int, default=1000)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument(
        "--threshold",
        type=float,
        default=-np.inf,
        help="Default disables score filtering so this run measures recall.",
    )
    parser.add_argument("--fredholm-radius", type=float)
    parser.add_argument("--template-side", type=int)
    parser.add_argument("--nms-radius", type=float)
    parser.add_argument("--score-template-batch-size", type=int)
    parser.add_argument("--score-memory-fraction", type=float, default=0.8)
    parser.add_argument(
        "--score-fft-shape",
        type=int,
        nargs=3,
        metavar=("Z", "Y", "X"),
        help="Override automatic cuFFT-friendly scoring dimensions.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU only for small debugging inputs.",
    )
    return parser.parse_args()


def configure_logging(log_file: Path) -> None:
    """Log every pipeline transition to stderr and a persistent file."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)


def mrc_voxel_size(
    path: Path,
    override: float | None,
) -> tuple[float, tuple[float, float, float]]:
    """Return isotropic spacing and the original ``(z, y, x)`` header values."""
    try:
        import mrcfile
    except ImportError as error:
        raise RuntimeError("mrcfile is required for this experiment") from error

    with mrcfile.open(path, mode="r", permissive=True, header_only=True) as mrc:
        spacing = (
            float(mrc.voxel_size.z),
            float(mrc.voxel_size.y),
            float(mrc.voxel_size.x),
        )
    if override is not None:
        if override <= 0:
            raise ValueError("--voxel-size must be positive")
        return override, spacing
    values = np.asarray(spacing)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError(f"invalid MRC spacing {spacing}; pass --voxel-size")
    if not np.allclose(values, values[0], rtol=1e-4):
        raise ValueError(f"anisotropic MRC spacing {spacing}; resample first")
    return float(values.mean()), spacing


def local_devices(*, allow_cpu: bool) -> tuple[jax.Device, ...]:
    """Use all local GPUs and guard against accidental login-node execution."""
    if jax.process_count() != 1:
        raise RuntimeError(
            "the multi-GPU processor expects one JAX process; found "
            f"{jax.process_count()}"
        )
    visible = tuple(jax.local_devices())
    gpus = tuple(device for device in visible if device.platform == "gpu")
    if gpus:
        return gpus
    if allow_cpu:
        return visible
    raise RuntimeError("JAX sees no GPU; use a GPU node or pass --allow-cpu")


def memory_limit_gib(device: jax.Device) -> float | None:
    """Return the allocator limit when exposed by the backend."""
    try:
        statistics = device.memory_stats()
    except (RuntimeError, NotImplementedError):
        return None
    if not statistics:
        return None
    limit = statistics.get(
        "bytes_limit",
        statistics.get("bytes_reservable_limit"),
    )
    return None if limit is None else int(limit) / 2**30


def load_ground_truth(path: Path) -> np.ndarray:
    """Load deposited ``(x, y, z)`` coordinates and return ``(z, y, x)``."""
    coordinates_xyz = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if coordinates_xyz.ndim != 2 or coordinates_xyz.shape[1] != 3:
        raise ValueError("ground truth must contain x, y, z columns")
    if not np.all(np.isfinite(coordinates_xyz)):
        raise ValueError("ground truth contains non-finite coordinates")
    return coordinates_xyz[:, ::-1].copy()


def _atomic_path(path: Path) -> tuple[Path, Any]:
    """Create a temporary binary stream beside its final checkpoint path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = tempfile.NamedTemporaryFile(
        mode="w+b",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    return Path(stream.name), stream


def save_pickle(value: object, path: Path) -> None:
    """Atomically save an arbitrary Python checkpoint."""
    temporary, stream = _atomic_path(path)
    try:
        with stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_npy(value: np.ndarray, path: Path) -> None:
    """Atomically save a NumPy array without an implicit filename suffix."""
    temporary, stream = _atomic_path(path)
    try:
        with stream:
            np.save(stream, value, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_npz(path: Path, **arrays: npt.ArrayLike) -> None:
    """Atomically save named analysis arrays."""
    temporary, stream = _atomic_path(path)
    try:
        with stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_json(value: object, path: Path) -> None:
    """Atomically save human-readable JSON."""
    temporary, stream = _atomic_path(path)
    try:
        with stream:
            stream.write(
                json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
            )
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_csv(
    value: np.ndarray,
    path: Path,
    *,
    header: str,
) -> None:
    """Atomically save a coordinate table."""
    temporary, stream = _atomic_path(path)
    try:
        with stream:
            np.savetxt(stream, value, delimiter=",", header=header, comments="")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_pickle(path: Path) -> Any:
    """Load a trusted local pipeline checkpoint."""
    with path.open("rb") as stream:
        return pickle.load(stream)


def prepare_block_qr_checkpoint(
    detector: MultiGPUKLTParticleDetector3D,
    templates: npt.ArrayLike,
    noise_variance: float,
    path: Path,
) -> None:
    """Build the large block-QR basis directly into an atomic NPY file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_stream = tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(temporary_stream.name)
    temporary_stream.close()
    try:
        template_array = np.asanyarray(templates)
        if detector.model.template_m_values is None:
            raise RuntimeError("template m values have not been initialized")
        representative_count = int(
            np.count_nonzero(detector.model.template_m_values >= 0)
        )
        output = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.complex64,
            shape=(representative_count, *template_array.shape[1:]),
        )
        detector.prepare_score_filters(
            template_array,
            noise_variance,
            output=output,
        )
        output.flush()
        detector.score_templates = None
        del output
        os.replace(temporary, path)
        detector.score_templates = np.load(
            path,
            mmap_mode="r",
            allow_pickle=False,
        )
    finally:
        if temporary.exists():
            temporary.unlink()


def run_stage(name: str, function: Callable[[], T]) -> tuple[T, float]:
    """Run one named stage with visible start, success, and timing logs."""
    LOGGER.info("=" * 72)
    LOGGER.info("STAGE START | %s", name)
    started = time.perf_counter()
    try:
        value = function()
    except Exception:
        LOGGER.exception("STAGE FAILED | %s", name)
        raise
    elapsed = time.perf_counter() - started
    LOGGER.info("STAGE DONE  | %s | %.2f minutes", name, elapsed / 60)
    return value, elapsed


def record_stage_time(
    stage_times: dict[str, float],
    name: str,
    elapsed: float,
) -> None:
    """Retain original timings when a later invocation resumes checkpoints."""
    if elapsed > 0 or name not in stage_times:
        stage_times[name] = elapsed


def require_replaceable(paths: tuple[Path, ...], *, overwrite: bool) -> None:
    """Reject accidental replacement of any completed stage artifact."""
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"{existing[0]} already exists; pass --resume or --overwrite"
        )


def checkpointed_stage(
    name: str,
    path: Path,
    function: Callable[[], T],
    *,
    resume: bool,
    overwrite: bool,
) -> tuple[T, float]:
    """Load a completed stage when resuming, otherwise run and checkpoint it."""
    if resume and path.is_file():
        LOGGER.info("STAGE RESUME | %s | loading %s", name, path)
        return load_pickle(path), 0.0
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists; pass --resume or --overwrite"
        )
    value, elapsed = run_stage(name, function)
    LOGGER.info("Checkpointing %s", path)
    save_pickle(value, path)
    return value, elapsed


def log_array(name: str, value: npt.ArrayLike) -> None:
    """Log shape, dtype, range, and finite status for an analysis array."""
    array = np.asanyarray(value)
    finite_count = 0
    minimum = np.inf
    maximum = -np.inf
    chunks = (
        (array[index] for index in range(array.shape[0]))
        if array.ndim and array.nbytes > 256 * 2**20
        else (array,)
    )
    for chunk in chunks:
        chunk_array = np.asarray(chunk)
        finite = np.isfinite(chunk_array)
        count = int(np.sum(finite))
        finite_count += count
        if count:
            real_values = np.real(chunk_array[finite])
            minimum = min(minimum, float(np.min(real_values)))
            maximum = max(maximum, float(np.max(real_values)))
    if array.size and finite_count:
        LOGGER.info(
            "%s | shape=%s dtype=%s finite=%d/%d range=[%.6g, %.6g]",
            name,
            array.shape,
            array.dtype,
            finite_count,
            array.size,
            minimum,
            maximum,
        )
    else:
        LOGGER.info(
            "%s | shape=%s dtype=%s | no finite values",
            name,
            array.shape,
            array.dtype,
        )


def validate_initial_rpsds(
    extraction: object,
    detector: MultiGPUKLTParticleDetector3D,
) -> None:
    """Reject a seed checkpoint that does not match this tomogram geometry."""
    required = ("rpsds", "variances", "patch_grid_shape", "patch_size")
    if any(not hasattr(extraction, name) for name in required):
        raise ValueError("initial RPSD checkpoint has an unsupported format")
    expected_grid = tuple(
        size // detector.patch_size for size in detector.source.shape
    )
    expected_count = math.prod(expected_grid)
    if extraction.patch_size != detector.patch_size:
        raise ValueError(
            "initial RPSD patch size does not match the current configuration"
        )
    if tuple(extraction.patch_grid_shape) != expected_grid:
        raise ValueError("initial RPSD grid does not match the current tomogram")
    if extraction.rpsds.shape[0] != expected_count:
        raise ValueError("initial RPSD sample count does not match its patch grid")
    if extraction.variances.shape != (expected_count,):
        raise ValueError("initial RPSD variance count is inconsistent")


def evaluate_recall(
    particles_zyx_score: np.ndarray,
    truth_zyx: np.ndarray,
    match_radius_voxels: float,
) -> tuple[dict[str, Any], np.ndarray]:
    """Compute maximum-cardinality one-to-one recall within a spatial radius."""
    predicted = np.asarray(particles_zyx_score, dtype=np.float64)
    if predicted.ndim != 2 or predicted.shape[1] != 4:
        raise ValueError("particles must contain z, y, x, score columns")
    if match_radius_voxels <= 0:
        raise ValueError("match radius must be positive")
    if predicted.shape[0] == 0 or truth_zyx.shape[0] == 0:
        matched = np.empty((0, 9), dtype=np.float64)
    else:
        distances = cdist(predicted[:, :3], truth_zyx)
        within_radius = distances <= match_radius_voxels
        predicted_indices, truth_indices = linear_sum_assignment(
            ~within_radius
        )
        accepted = within_radius[predicted_indices, truth_indices]
        predicted_indices = predicted_indices[accepted]
        truth_indices = truth_indices[accepted]
        matched = np.column_stack(
            (
                predicted_indices,
                truth_indices,
                predicted[predicted_indices, :3],
                truth_zyx[truth_indices],
                distances[predicted_indices, truth_indices],
            )
        )
    matched_count = matched.shape[0]
    summary = {
        "ground_truth_count": int(truth_zyx.shape[0]),
        "requested_pick_count": int(truth_zyx.shape[0]),
        "returned_pick_count": int(predicted.shape[0]),
        "matched_ground_truth_count": int(matched_count),
        "recall": float(matched_count / truth_zyx.shape[0]),
        "match_radius_voxels": float(match_radius_voxels),
        "mean_matched_distance_voxels": (
            None if not matched_count else float(np.mean(matched[:, -1]))
        ),
        "median_matched_distance_voxels": (
            None if not matched_count else float(np.median(matched[:, -1]))
        ),
    }
    return summary, matched


def log_plan(
    args: argparse.Namespace,
    detector: MultiGPUKLTParticleDetector3D,
    voxel_size: float,
    header_spacing: tuple[float, float, float],
    truth_count: int,
) -> None:
    """Log the complete resolved geometry before expensive compilation."""
    processor = detector.processor
    patch_grid = tuple(size // detector.patch_size for size in detector.source.shape)
    core_patch_shape = tuple(
        size // detector.patch_size for size in processor.core_shape
    )
    whitening_halo = detector.whitening_support_radius
    template_radius = detector.model.template_side // 2
    scoring_halo = whitening_halo + template_radius + 1
    LOGGER.info("Input: %s | %.2f GiB", args.input, args.input.stat().st_size / 2**30)
    LOGGER.info(
        "Volume: shape(z,y,x)=%s | voxel=%.7f A | header=%s",
        detector.source.shape,
        voxel_size,
        tuple(round(value, 7) for value in header_spacing),
    )
    LOGGER.info(
        "Truth: %s | particles=%d | NMS requested picks=%d",
        args.ground_truth,
        truth_count,
        detector.model.num_particles,
    )
    LOGGER.info(
        "RPSD: patch=%d^3 | grid=%s | total patches=%d | microbatch=%d",
        detector.patch_size,
        patch_grid,
        math.prod(patch_grid),
        detector.patches_per_microbatch,
    )
    LOGGER.info(
        "KLT: Legendre=%d | angular orders=%d (ell=0..%d) | "
        "energy=%.3f | template cap=%s | Fredholm radius=%.2f | template=%d^3",
        detector.model.legendre_order,
        detector.model.max_order,
        detector.model.max_order - 1,
        detector.model.template_energy_fraction,
        detector.model.max_templates,
        detector.model.fredholm_radius_voxels,
        detector.model.template_side,
    )
    LOGGER.info(
        "Preprocessing: finite band-pass removes low/high %.1f%%/%.1f%% | "
        "support=%d | score basis=(ell,m) block QR",
        100 * detector.bandpass_low_fraction,
        100 * detector.bandpass_high_fraction,
        whitening_halo,
    )
    LOGGER.info(
        "Halos: whitening=%d | template=%d | local-max=1 | scoring total=%d",
        whitening_halo,
        template_radius,
        scoring_halo,
    )
    LOGGER.info(
        "Per GPU: core patches=%s | core voxels=%s | scoring load=%s",
        core_patch_shape,
        processor.core_shape,
        processor.loaded_shape(scoring_halo),
    )
    LOGGER.info(
        "Schedule: subvolume grid=%s | RPSD rounds=%d | scoring subvolumes=%d | "
        "single scoring input=%.2f GiB",
        processor.subvolume_grid_shape,
        processor.round_count,
        processor.subvolume_count,
        int(np.prod(processor.loaded_shape(scoring_halo)))
        * np.dtype(np.float32).itemsize
        / 2**30,
    )
    LOGGER.info(
        "Candidates: static capacity/subvolume=%d | threshold=%s | NMS radius=%.2f",
        detector.candidate_capacity_per_subvolume,
        detector.model.threshold,
        detector.model.nms_radius_voxels,
    )


def main() -> None:
    """Run the complete checkpointed experiment and report recall."""
    args = parse_args()
    args.results_dir = args.results_dir.resolve()
    log_file = (
        args.results_dir / "empiar-10045.log"
        if args.log_file is None
        else args.log_file.resolve()
    )
    configure_logging(log_file)
    started = time.perf_counter()
    stage_times: dict[str, float] = {}
    LOGGER.info(
        "Experiment process started | pid=%d | results=%s",
        os.getpid(),
        args.results_dir,
    )
    LOGGER.info("Initializing input metadata and JAX devices")

    try:
        if not args.input.is_file():
            raise FileNotFoundError(args.input)
        if not args.ground_truth.is_file():
            raise FileNotFoundError(args.ground_truth)
        args.results_dir.mkdir(parents=True, exist_ok=True)
        previous_evaluation = args.results_dir / "09_evaluation.json"
        if args.resume and previous_evaluation.is_file():
            with previous_evaluation.open() as stream:
                previous = json.load(stream)
            stage_times.update(previous.get("stage_runtime_seconds", {}))

        voxel_size, header_spacing = mrc_voxel_size(args.input, args.voxel_size)
        truth_zyx = load_ground_truth(args.ground_truth)
        truth_count = truth_zyx.shape[0]
        match_radius_angstrom = (
            args.particle_diameter / 2
            if args.match_radius_angstrom is None
            else args.match_radius_angstrom
        )
        if match_radius_angstrom <= 0:
            raise ValueError("--match-radius-angstrom must be positive")
        match_radius_voxels = match_radius_angstrom / voxel_size

        devices = local_devices(allow_cpu=args.allow_cpu)
        LOGGER.info(
            "JAX %s | backend=%s | local devices=%d",
            jax.__version__,
            devices[0].platform,
            len(devices),
        )
        for index, device in enumerate(devices):
            limit = memory_limit_gib(device)
            LOGGER.info(
                "Device %d: %s | allocator limit=%s",
                index,
                device.device_kind,
                "unavailable" if limit is None else f"{limit:.2f} GiB",
            )

        with MrcVolumeSource(args.input) as source:
            detector = MultiGPUKLTParticleDetector3D(
                source,
                particle_diameter=args.particle_diameter,
                mgscale=1.0 / voxel_size,
                num_particles=truth_count,
                whitening_support_radius=args.whitening_support_radius,
                bandpass_low_fraction=args.bandpass_low_fraction,
                bandpass_high_fraction=args.bandpass_high_fraction,
                devices=devices,
                core_patch_shape=(
                    None
                    if args.core_patch_shape is None
                    else tuple(args.core_patch_shape)
                ),
                memory_fraction=args.memory_fraction,
                resident_volume_copies=args.resident_volume_copies,
                patches_per_microbatch=args.patches_per_microbatch,
                candidate_capacity_per_subvolume=(
                    args.candidate_capacity_per_subvolume
                ),
                legendre_order=args.legendre_order,
                threshold=args.threshold,
                max_iter=args.max_iterations,
                max_order=args.max_order,
                template_energy_fraction=args.template_energy_fraction,
                max_templates=args.max_templates,
                fredholm_radius=args.fredholm_radius,
                template_side=args.template_side,
                nms_radius=args.nms_radius,
                score_template_batch_size=args.score_template_batch_size,
                score_memory_fraction=args.score_memory_fraction,
                score_fft_shape=(
                    None
                    if args.score_fft_shape is None
                    else tuple(args.score_fft_shape)
                ),
            )
            log_plan(args, detector, voxel_size, header_spacing, truth_count)
            LOGGER.info(
                "Recall matching radius: %.2f A = %.2f voxels",
                match_radius_angstrom,
                match_radius_voxels,
            )

            manifest = {
                "input": str(args.input.resolve()),
                "ground_truth": str(args.ground_truth.resolve()),
                "volume_shape_zyx": source.shape,
                "voxel_size_angstrom": voxel_size,
                "particle_diameter_angstrom": args.particle_diameter,
                "truth_count": truth_count,
                "patch_size": detector.patch_size,
                "core_shape": detector.processor.core_shape,
                "devices": [device.device_kind for device in devices],
                "whitening_support_radius": args.whitening_support_radius,
                "bandpass_low_fraction": args.bandpass_low_fraction,
                "bandpass_high_fraction": args.bandpass_high_fraction,
                "score_basis": "distributed_block_qr_nonnegative_m_v3",
                "template_side": detector.model.template_side,
                "fredholm_radius_voxels": detector.model.fredholm_radius_voxels,
                "max_order": detector.model.max_order,
                "template_energy_fraction": args.template_energy_fraction,
                "max_templates": args.max_templates,
                "score_template_batch_size": args.score_template_batch_size,
                "score_memory_fraction": args.score_memory_fraction,
                "score_fft_shape": detector.score_fft_shape,
                "nms_radius_voxels": detector.model.nms_radius_voxels,
                "match_radius_angstrom": match_radius_angstrom,
                "match_radius_voxels": match_radius_voxels,
            }
            save_json(manifest, args.results_dir / "00_manifest.json")
            save_npy(truth_zyx, args.results_dir / "00_ground_truth_zyx.npy")
            save_csv(
                truth_zyx[:, ::-1],
                args.results_dir / "00_ground_truth_xyz.csv",
                header="x,y,z",
            )
            if args.dry_run:
                LOGGER.info("Dry run complete; no numerical stage was started.")
                return

            initial_path = args.results_dir / "01_initial_patch_rpsds.pkl"
            detector.bandpass_filter = detector.build_bandpass_filter()
            save_npy(
                detector.bandpass_filter,
                args.results_dir / "00_bandpass_filter.npy",
            )
            LOGGER.info(
                "Finite band-pass: low=%.3f high=%.3f support=%d L2=%.8g",
                args.bandpass_low_fraction,
                args.bandpass_high_fraction,
                args.whitening_support_radius,
                np.linalg.norm(detector.bandpass_filter),
            )
            if args.resume and initial_path.is_file():
                LOGGER.info(
                    "STAGE RESUME | initial RPSDs | loading %s",
                    initial_path,
                )
                detector.initial_rpsds = load_pickle(initial_path)
                record_stage_time(stage_times, "initial_rpsds", 0.0)
            elif (
                not args.recompute_initial
                and args.initial_rpsds is not None
                and args.initial_rpsds.is_file()
            ):
                if initial_path.exists() and not args.overwrite:
                    raise FileExistsError(
                        f"{initial_path} exists; pass --resume or --overwrite"
                    )
                LOGGER.info(
                    "STAGE SEED | initial RPSDs | loading existing %s",
                    args.initial_rpsds,
                )
                detector.initial_rpsds = load_pickle(args.initial_rpsds)
                save_pickle(detector.initial_rpsds, initial_path)
                record_stage_time(stage_times, "initial_rpsds", 0.0)
            else:
                detector.initial_rpsds, elapsed = checkpointed_stage(
                    "1/8 initial streamed patch RPSDs",
                    initial_path,
                    lambda: detector.estimate_rpsds(
                        detector.bandpass_filter,
                        description="Band-passed RPSD extraction",
                    ),
                    resume=args.resume,
                    overwrite=args.overwrite,
                )
                record_stage_time(stage_times, "initial_rpsds", elapsed)
            validate_initial_rpsds(detector.initial_rpsds, detector)
            log_array("Initial patch RPSDs", detector.initial_rpsds.rpsds)
            log_array("Initial patch variances", detector.initial_rpsds.variances)

            detector.initial_model, elapsed = checkpointed_stage(
                "2/8 initial ALS and variance calibration",
                args.results_dir / "02_initial_als.pkl",
                lambda: detector.fit_rpsds(detector.initial_rpsds),
                resume=args.resume,
                overwrite=args.overwrite,
            )
            record_stage_time(stage_times, "initial_als", elapsed)
            LOGGER.info(
                "Initial ALS: noise variance=%.8g",
                detector.initial_model.noise_variance,
            )
            log_array("Initial particle PSD", detector.initial_model.particle_psd)
            log_array("Initial noise PSD", detector.initial_model.noise_psd)

            detector.whitening_filter, elapsed = checkpointed_stage(
                "3/8 finite combined band-pass/whitening filter construction",
                args.results_dir / "03_whitening_filter.pkl",
                lambda: detector.build_whitening_filter(
                    detector.initial_model.noise_psd
                ),
                resume=args.resume,
                overwrite=args.overwrite,
            )
            record_stage_time(stage_times, "whitening_filter", elapsed)
            save_npy(
                detector.whitening_filter,
                args.results_dir / "03_whitening_filter.npy",
            )
            log_array("Whitening filter", detector.whitening_filter)
            LOGGER.info(
                "Whitening filter L2 norm=%.8g",
                np.linalg.norm(detector.whitening_filter),
            )

            detector.whitened_rpsds, elapsed = checkpointed_stage(
                "4/8 whitened streamed patch RPSDs",
                args.results_dir / "04_whitened_patch_rpsds.pkl",
                lambda: detector.estimate_rpsds(
                    detector.whitening_filter,
                    description="Band-passed whitened RPSD extraction",
                ),
                resume=args.resume,
                overwrite=args.overwrite,
            )
            record_stage_time(stage_times, "whitened_rpsds", elapsed)
            log_array("Whitened patch RPSDs", detector.whitened_rpsds.rpsds)
            log_array(
                "Whitened patch variances",
                detector.whitened_rpsds.variances,
            )

            detector.whitened_model, elapsed = checkpointed_stage(
                "5/8 whitened ALS and variance calibration",
                args.results_dir / "05_whitened_als.pkl",
                lambda: detector.fit_rpsds(detector.whitened_rpsds),
                resume=args.resume,
                overwrite=args.overwrite,
            )
            record_stage_time(stage_times, "whitened_als", elapsed)
            LOGGER.info(
                "Whitened ALS: noise variance=%.8g",
                detector.whitened_model.noise_variance,
            )
            log_array(
                "Whitened particle PSD",
                detector.whitened_model.particle_psd,
            )
            log_array("Whitened noise PSD", detector.whitened_model.noise_psd)

            templates_path = args.results_dir / "06_templates.npy"
            template_metadata_path = (
                args.results_dir / "06_template_metadata.npz"
            )
            template_checkpoint_compatible = False
            if template_metadata_path.is_file():
                with np.load(template_metadata_path, allow_pickle=False) as metadata:
                    template_checkpoint_compatible = (
                        "inverse_fourier_normalization_3d" in metadata
                        and np.isclose(
                            metadata["inverse_fourier_normalization_3d"].item(),
                            INVERSE_FOURIER_NORMALIZATION_3D,
                        )
                    )
            templates_recomputed = False
            if (
                args.resume
                and templates_path.is_file()
                and template_metadata_path.is_file()
                and template_checkpoint_compatible
            ):
                LOGGER.info(
                    "STAGE RESUME | templates | loading %s",
                    templates_path,
                )
                detector.templates = np.load(
                    templates_path,
                    mmap_mode="r",
                    allow_pickle=False,
                )
                elapsed = 0.0
            else:
                if (
                    args.resume
                    and templates_path.is_file()
                    and template_metadata_path.is_file()
                    and not template_checkpoint_compatible
                ):
                    if not args.overwrite:
                        raise RuntimeError(
                            "stage-6 checkpoint uses the legacy Fourier scale; "
                            "pass --resume --overwrite to rebuild stage 6 onward"
                        )
                    LOGGER.warning(
                        "Stage-6 checkpoint predates corrected Fourier "
                        "normalization; rebuilding templates and eigenvalues"
                    )
                require_replaceable(
                    (templates_path, template_metadata_path),
                    overwrite=args.overwrite,
                )
                templates_recomputed = True
                detector.templates, elapsed = run_stage(
                    "6/8 Fredholm solve and KLT template construction",
                    lambda: detector.build_templates(
                        detector.whitened_model.particle_psd
                    ),
                )
                save_npy(detector.templates, templates_path)
                detector.templates = np.load(
                    templates_path,
                    mmap_mode="r",
                    allow_pickle=False,
                )
                save_npz(
                    template_metadata_path,
                    template_eigenvalues=detector.model.eigvals,
                    radial_eigenvalues=detector.model.radial_eigvals,
                    radial_eigenfunctions=detector.model.eigfuncs,
                    template_orders=detector.model.template_orders,
                    template_m_values=detector.model.template_m_values,
                    template_multiplicities=(
                        detector.model.template_multiplicities
                    ),
                    available_radial_mode_count=np.asarray(
                        detector.model.available_radial_mode_count
                    ),
                    available_template_count=np.asarray(
                        detector.model.available_template_count
                    ),
                    retained_radial_mode_count=np.asarray(
                        detector.model.retained_radial_mode_count
                    ),
                    retained_template_count=np.asarray(
                        detector.model.retained_template_count
                    ),
                    retained_template_energy_fraction=np.asarray(
                        detector.model.retained_template_energy_fraction
                    ),
                    template_energy_fraction_config=np.asarray(
                        detector.model.template_energy_fraction
                    ),
                    max_order_config=np.asarray(detector.model.max_order),
                    max_templates_config=np.asarray(
                        -1
                        if detector.model.max_templates is None
                        else detector.model.max_templates
                    ),
                    inverse_fourier_normalization_3d=np.asarray(
                        INVERSE_FOURIER_NORMALIZATION_3D
                    ),
                )
            record_stage_time(stage_times, "templates", elapsed)
            with np.load(template_metadata_path, allow_pickle=False) as metadata:
                detector.model.eigvals = metadata[
                    "template_eigenvalues"
                ].copy()
                detector.model.radial_eigvals = metadata[
                    "radial_eigenvalues"
                ].copy()
                detector.model.eigfuncs = metadata[
                    "radial_eigenfunctions"
                ].copy()
                detector.model.template_orders = metadata[
                    "template_orders"
                ].copy()
                detector.model.template_m_values = metadata[
                    "template_m_values"
                ].copy()
                detector.model.template_multiplicities = (
                    metadata["template_multiplicities"].copy()
                    if "template_multiplicities" in metadata
                    else (
                        np.ones_like(
                            detector.model.template_m_values,
                            dtype=np.float32,
                        )
                        if np.any(detector.model.template_m_values < 0)
                        else np.where(
                            detector.model.template_m_values == 0,
                            1,
                            2,
                        ).astype(np.float32)
                    )
                )
                for name in (
                    "available_radial_mode_count",
                    "available_template_count",
                    "retained_radial_mode_count",
                    "retained_template_count",
                    "retained_template_energy_fraction",
                ):
                    if name in metadata:
                        setattr(detector.model, name, metadata[name].item())
            if detector.model.eigvals.shape != (detector.templates.shape[0],):
                raise RuntimeError(
                    "template checkpoint metadata does not match the template "
                    "array; rerun stage 6 with --overwrite"
                )
            if np.any(detector.model.template_orders >= detector.model.max_order):
                raise RuntimeError(
                    "template checkpoint contains angular orders excluded by "
                    "the current --max-order; rerun stage 6 with --overwrite"
                )
            if (
                detector.model.max_templates is not None
                and detector.templates.shape[0] > detector.model.max_templates
            ):
                raise RuntimeError(
                    "template checkpoint exceeds the current --max-templates; "
                    "rerun stage 6 with --overwrite"
                )
            with np.load(template_metadata_path, allow_pickle=False) as metadata:
                if (
                    "template_energy_fraction_config" in metadata
                    and not np.isclose(
                        metadata["template_energy_fraction_config"].item(),
                        detector.model.template_energy_fraction,
                    )
                ):
                    raise RuntimeError(
                        "template checkpoint uses a different energy fraction; "
                        "rerun stage 6 with --overwrite"
                    )
                if (
                    "max_order_config" in metadata
                    and metadata["max_order_config"].item()
                    != detector.model.max_order
                ):
                    raise RuntimeError(
                        "template checkpoint uses a different max_order; "
                        "rerun stage 6 with --overwrite"
                    )
                configured_cap = (
                    -1
                    if detector.model.max_templates is None
                    else detector.model.max_templates
                )
                if (
                    "max_templates_config" in metadata
                    and metadata["max_templates_config"].item() != configured_cap
                ):
                    raise RuntimeError(
                        "template checkpoint uses a different template cap; "
                        "rerun stage 6 with --overwrite"
                    )
            log_array("KLT templates", detector.templates)
            log_array("Template eigenvalues", detector.model.eigvals)
            LOGGER.info(
                "Templates: stored m>=0 representatives=%d | effective complete "
                "modes=%s | radial modes=%s | available complete=%s | "
                "retained energy=%s | spatial shape=%s",
                detector.templates.shape[0],
                detector.model.retained_template_count,
                detector.model.retained_radial_mode_count,
                detector.model.available_template_count,
                (
                    "unknown"
                    if detector.model.retained_template_energy_fraction is None
                    else f"{detector.model.retained_template_energy_fraction:.6f}"
                ),
                detector.templates.shape[1:],
            )

            score_templates_path = args.results_dir / "06b_block_qr_templates.npy"
            score_model_path = args.results_dir / "06b_block_qr_score_model.npz"
            score_checkpoint_compatible = False
            if score_model_path.is_file():
                with np.load(score_model_path, allow_pickle=False) as score_model:
                    score_checkpoint_compatible = (
                        "method" in score_model
                        and score_model["method"].item() == _SCORE_MODEL_METHOD
                        and "score_multiplicities" in score_model
                        and "score_template_indices" in score_model
                    )
            score_model_recomputed = False
            if (
                args.resume
                and not templates_recomputed
                and score_templates_path.is_file()
                and score_model_path.is_file()
                and score_checkpoint_compatible
            ):
                LOGGER.info(
                    "STAGE RESUME | (ell,m) block-QR score model | loading %s",
                    score_model_path,
                )
                detector.score_templates = np.load(
                    score_templates_path,
                    mmap_mode="r",
                    allow_pickle=False,
                )
                with np.load(score_model_path, allow_pickle=False) as score_model:
                    if (
                        "noise_variance" in score_model
                        and not np.isclose(
                            score_model["noise_variance"].item(),
                            detector.whitened_model.noise_variance,
                        )
                    ):
                        raise RuntimeError(
                            "score-model checkpoint uses a different noise "
                            "variance; rerun stage 6b with --overwrite"
                        )
                    detector.template_normalization = score_model[
                        "template_normalization"
                    ].copy()
                    detector.score_weights = score_model["score_weights"].copy()
                    detector.score_offset = np.float32(score_model["score_offset"])
                    detector.adjusted_template_eigenvalues = score_model[
                        "adjusted_template_eigenvalues"
                    ].copy()
                    detector.score_multiplicities = score_model[
                        "score_multiplicities"
                    ].copy()
                    detector.score_template_indices = score_model[
                        "score_template_indices"
                    ].copy()
            else:
                if (
                    args.resume
                    and score_model_path.is_file()
                    and not score_checkpoint_compatible
                ):
                    if not args.overwrite:
                        raise RuntimeError(
                            "stage-6b checkpoint uses the legacy likelihood "
                            "scale; pass --resume --overwrite to rebuild stage "
                            "6b onward"
                        )
                    LOGGER.warning(
                        "Stage-6b checkpoint uses the legacy likelihood scale; "
                        "rebuilding block-QR score parameters"
                    )
                require_replaceable(
                    (score_templates_path, score_model_path),
                    overwrite=args.overwrite,
                )
                score_model_recomputed = True
                run_stage(
                    "6b/8 distributed (ell,m) block QR and score model",
                    lambda: prepare_block_qr_checkpoint(
                        detector,
                        detector.templates,
                        detector.whitened_model.noise_variance,
                        score_templates_path,
                    ),
                )
                save_npz(
                    score_model_path,
                    method=np.asarray(_SCORE_MODEL_METHOD),
                    template_normalization=detector.template_normalization,
                    score_weights=detector.score_weights,
                    score_offset=np.asarray(detector.score_offset),
                    adjusted_template_eigenvalues=(
                        detector.adjusted_template_eigenvalues
                    ),
                    score_multiplicities=detector.score_multiplicities,
                    score_template_indices=detector.score_template_indices,
                    noise_variance=np.asarray(
                        detector.whitened_model.noise_variance
                    ),
                )
            log_array("KLT score weights", detector.score_weights)
            LOGGER.info("Block-QR score templates: %s", score_templates_path)
            LOGGER.info(
                "Conjugate symmetry: executed representatives=%d | "
                "effective modes=%d",
                detector.score_templates.shape[0],
                int(np.sum(detector.score_multiplicities)),
            )
            LOGGER.info("KLT likelihood offset: %.8g", detector.score_offset)

            candidate_tag = f"top{args.candidate_capacity_per_subvolume}"
            candidates_path = args.results_dir / f"07_candidates_{candidate_tag}.npy"
            score_plan_path = args.results_dir / f"07_score_plan_{candidate_tag}.json"
            candidates_recomputed = False
            if (
                args.resume
                and not score_model_recomputed
                and candidates_path.is_file()
            ):
                LOGGER.info(
                    "STAGE RESUME | scoring candidates | loading %s",
                    candidates_path,
                )
                detector.candidates = np.load(candidates_path, allow_pickle=False)
                elapsed = 0.0
            else:
                require_replaceable(
                    (candidates_path, score_plan_path),
                    overwrite=args.overwrite,
                )
                detector.candidates, elapsed = run_stage(
                    "7/8 fused whitening, KLT scoring, and local maxima",
                    lambda: detector.score_candidates(
                        detector.score_templates,
                        detector.whitened_model.noise_variance,
                        detector.whitening_filter,
                    ),
                )
                save_npy(detector.candidates, candidates_path)
                candidates_recomputed = True
                if detector.score_plan is not None:
                    save_json(detector.score_plan, score_plan_path)
            record_stage_time(stage_times, "scoring", elapsed)
            LOGGER.info(
                "Candidates returned from all GPUs: %d",
                len(detector.candidates),
            )
            if len(detector.candidates):
                log_array("Candidate raw scores", detector.candidates[:, 3])

            particles_path = (
                args.results_dir / f"08_particles_{candidate_tag}_zyx.npy"
            )
            if (
                args.resume
                and not candidates_recomputed
                and particles_path.is_file()
            ):
                LOGGER.info(
                    "STAGE RESUME | global NMS | loading %s",
                    particles_path,
                )
                detector.particles = np.load(particles_path, allow_pickle=False)
                elapsed = 0.0
            else:
                require_replaceable(
                    (particles_path,),
                    overwrite=args.overwrite,
                )
                detector.particles, elapsed = run_stage(
                    "8/8 global ranking and NMS",
                    lambda: detector.non_maximum_suppression(
                        detector.candidates
                    ),
                )
                save_npy(detector.particles, particles_path)
            record_stage_time(stage_times, "global_nms", elapsed)
            particles_xyz_score = detector.particles[:, [2, 1, 0, 3]]
            save_csv(
                particles_xyz_score,
                args.results_dir / "08_particles_xyz.csv",
                header="x,y,z,normalized_score",
            )
            LOGGER.info(
                "NMS returned %d coordinates (requested %d)",
                len(detector.particles),
                truth_count,
            )

        evaluation, matches = evaluate_recall(
            detector.particles,
            truth_zyx,
            match_radius_voxels,
        )
        evaluation["match_radius_angstrom"] = float(match_radius_angstrom)
        evaluation["total_runtime_minutes"] = (
            time.perf_counter() - started
        ) / 60
        evaluation["stage_runtime_seconds"] = stage_times
        save_json(evaluation, args.results_dir / "09_evaluation.json")
        save_csv(
            matches,
            args.results_dir / "09_matches.csv",
            header=(
                "prediction_index,truth_index,pred_z,pred_y,pred_x,"
                "truth_z,truth_y,truth_x,distance_voxels"
            ),
        )
        save_pickle(
            {
                "particles_zyx_score": detector.particles,
                "ground_truth_zyx": truth_zyx,
                "matches": matches,
                "evaluation": evaluation,
            },
            args.results_dir / "09_final_result.pkl",
        )
        LOGGER.info("=" * 72)
        LOGGER.info(
            "FINAL RECALL | matched=%d / truth=%d | recall=%.4f",
            evaluation["matched_ground_truth_count"],
            evaluation["ground_truth_count"],
            evaluation["recall"],
        )
        LOGGER.info(
            "Coordinates: %s",
            args.results_dir / "08_particles_xyz.csv",
        )
        LOGGER.info(
            "Experiment completed in %.2f minutes | results=%s",
            evaluation["total_runtime_minutes"],
            args.results_dir,
        )
    except Exception:
        LOGGER.exception(
            "Experiment failed after %.2f minutes",
            (time.perf_counter() - started) / 60,
        )
        raise


if __name__ == "__main__":
    main()
