"""Profile steady-state template-sharded EMPIAR-10045 scoring.

The harness loads the existing whitening and block-QR checkpoints, constructs
the same ``pmap`` and template shards as the production scorer, warms up JIT
compilation and cuFFT planning, and then profiles a bounded number of repeated
executions on one real haloed subvolume. It does not run RPSD estimation, ALS,
Fredholm solves, QR, global NMS, or the full tomogram.

The outer NVTX range is named ``empiar_scoring_profile`` so Nsight Systems can
start collection only after warm-up. JAX/Perfetto tracing is optional and
should be run separately from Nsight Systems because both use GPU profiling
facilities.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import ctypes.util
import json
import logging
import math
import sys
import time
from collections.abc import Iterator, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import jax
import numpy as np
import numpy.typing as npt


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from kltpicker_3d.multi_gpu import (
    _TEMPLATE_AXIS_NAME,
    plan_cufft_fft_shape,
    score_template_shards_and_extract_candidates,
)
from kltpicker_3d.streaming import (
    MrcVolumeSource,
    MultiGPUSubvolumeProcessor,
    spatial_filter_radius,
)


LOGGER = logging.getLogger("profile-empiar-scoring")
DEFAULT_RESULTS_DIR = REPOSITORY_ROOT / "results/empiar-10045-bandpass-block-qr"
DEFAULT_PROFILE_DIR = REPOSITORY_ROOT / "results/profiles/empiar-current-scoring"
TEMPLATE_AXIS_NAME = _TEMPLATE_AXIS_NAME
NVTX_CAPTURE_RANGE = "empiar_scoring_profile"
CLI_EPILOG = f"""
examples:
  Validate checkpoints without allocating GPU shards:
    python experiments/profile_empiar_scoring.py --dry-run

  Write a JAX/Perfetto trace (run separately from Nsight Systems):
    python experiments/profile_empiar_scoring.py --jax-trace

  Nsight Systems should use this capture configuration around the script:
    nsys profile --trace=cuda,nvtx,osrt --capture-range=nvtx \\
      --nvtx-capture={NVTX_CAPTURE_RANGE} --capture-range-end=stop \\
      --cuda-memory-usage=true --sample=none --stats=true \\
      --output=results/profiles/empiar-current-scoring/nsight \\
      python -u experiments/profile_empiar_scoring.py --require-nvtx
"""


def parse_args() -> argparse.Namespace:
    """Parse checkpoint, geometry, and profiling options."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=CLI_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--input",
        type=Path,
        help="MRC override; default uses the path recorded in 00_manifest.json.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PROFILE_DIR)
    parser.add_argument(
        "--device-count",
        type=int,
        help="Use the first N visible devices; default uses every visible GPU.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Permit CPU only for dry-run/CI validation; Nsight requires GPUs.",
    )
    parser.add_argument(
        "--core-patch-shape",
        type=int,
        nargs=3,
        metavar=("Z", "Y", "X"),
        help="Override the checkpoint core geometry in units of checkpoint patches.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Template FFT batch; default uses the saved stage-7 score plan.",
    )
    parser.add_argument("--candidate-capacity", type=int, default=4096)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--profile-iterations", type=int, default=3)
    parser.add_argument(
        "--region-index",
        type=int,
        help="Core-region traversal index; default selects the first interior core.",
    )
    parser.add_argument(
        "--jax-trace",
        action="store_true",
        help="Write an XPlane and Perfetto trace under <output-dir>/jax-trace.",
    )
    parser.add_argument(
        "--require-nvtx",
        action="store_true",
        help="Fail if the Nsight-injected NVTX capture range is inactive.",
    )
    parser.add_argument(
        "--skip-memory-profile",
        action="store_true",
        help="Do not write the post-warm-up JAX device-memory profile.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate checkpoints and print geometry without allocating shards.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    """Configure concise timestamps on the SSH terminal."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)


def require_positive(name: str, value: int) -> None:
    """Reject invalid positive integer settings."""
    if value < 1:
        raise ValueError(f"{name} must be positive")


def required_checkpoint(results_dir: Path, name: str) -> Path:
    """Return one required profiling checkpoint or fail descriptively."""
    path = results_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"required scoring checkpoint is missing: {path}")
    return path


def json_compatible(value: Any) -> Any:
    """Convert profiler metadata containing NumPy scalars into JSON values."""
    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_compatible(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class NvtxLibrary:
    """Minimal dependency-free NVTX wrapper for Nsight capture ranges."""

    def __init__(self, *, require_active: bool = False) -> None:
        library_name = ctypes.util.find_library("nvToolsExt")
        if library_name is None:
            raise RuntimeError("libnvToolsExt is unavailable")
        self._library = ctypes.CDLL(library_name)
        self._library.nvtxRangePushA.argtypes = [ctypes.c_char_p]
        self._library.nvtxRangePushA.restype = ctypes.c_int
        self._library.nvtxRangePop.argtypes = []
        self._library.nvtxRangePop.restype = ctypes.c_int
        self._require_active = require_active
        self._warned_inactive = False

    @contextlib.contextmanager
    def range(self, name: str) -> Iterator[None]:
        """Push and pop one thread-local NVTX range."""
        result = self._library.nvtxRangePushA(name.encode("utf-8"))
        if result < 0:
            if self._require_active:
                raise RuntimeError(
                    f"nvtxRangePushA is inactive for {name!r}; launch the script "
                    "through the documented nsys command"
                )
            if not self._warned_inactive:
                LOGGER.warning(
                    "NVTX ranges are inactive; timing and JAX profiling remain "
                    "available, but Nsight range capture requires nsys injection"
                )
                self._warned_inactive = True
            yield
            return
        try:
            yield
        finally:
            if self._library.nvtxRangePop() < 0:
                raise RuntimeError(f"nvtxRangePop failed for {name!r}")


def memory_stats(devices: Sequence[jax.Device]) -> list[dict[str, Any]]:
    """Return the allocator statistics currently exposed by every device."""
    statistics = []
    for device in devices:
        try:
            values = device.memory_stats()
        except (AttributeError, RuntimeError):
            values = None
        statistics.append(
            {
                "device": str(device),
                "stats": None if values is None else json_compatible(values),
            }
        )
    return statistics


def select_devices(
    device_count: int | None,
    *,
    allow_cpu: bool,
) -> tuple[jax.Device, ...]:
    """Select a deterministic prefix of visible local GPUs."""
    available = tuple(jax.local_devices())
    if not available:
        raise RuntimeError("JAX did not expose any local devices")
    if available[0].platform != "gpu" and not allow_cpu:
        raise RuntimeError(
            "profiling requires a GPU backend; check CUDA_VISIBLE_DEVICES and JAX"
        )
    if device_count is None:
        return available
    require_positive("device-count", device_count)
    if device_count > len(available):
        raise ValueError(
            f"requested {device_count} devices but only {len(available)} are visible"
        )
    return available[:device_count]


def checkpoint_core_shape(
    manifest: dict[str, Any],
    core_patch_shape: Sequence[int] | None,
) -> tuple[int, int, int]:
    """Resolve the exact owned core shape in voxels."""
    patch_size = int(manifest["patch_size"])
    if core_patch_shape is not None:
        if len(core_patch_shape) != 3 or any(value < 1 for value in core_patch_shape):
            raise ValueError("core-patch-shape must contain three positive values")
        return tuple(int(value) * patch_size for value in core_patch_shape)
    saved = tuple(int(value) for value in manifest["core_shape"])
    if len(saved) != 3 or any(value < 1 for value in saved):
        raise ValueError("manifest contains an invalid core_shape")
    return saved


def resolve_batch_size(
    requested: int | None,
    results_dir: Path,
    templates_per_device: int,
) -> int:
    """Use an explicit batch or reproduce the saved production score plan."""
    if requested is not None:
        require_positive("batch-size", requested)
        return min(requested, templates_per_device)
    plan_path = results_dir / "07_score_plan_top4096.json"
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text())
        batch_size = int(plan["batch_size"])
        require_positive("saved batch_size", batch_size)
        return min(batch_size, templates_per_device)
    LOGGER.warning("No saved stage-7 score plan; falling back to batch size 1")
    return 1


def select_region(
    processor: MultiGPUSubvolumeProcessor,
    halo: int,
    region_index: int | None,
) -> tuple[int, Any]:
    """Select an explicit region or the first one requiring no source padding."""
    regions = list(processor.regions())
    if region_index is not None:
        if not 0 <= region_index < len(regions):
            raise IndexError(
                f"region-index {region_index} is outside [0, {len(regions)})"
            )
        return region_index, regions[region_index]
    source_shape = processor.source.shape
    for index, region in enumerate(regions):
        if all(
            start >= halo and stop + halo <= source_size
            for start, stop, source_size in zip(
                region.start,
                region.stop,
                source_shape,
                strict=True,
            )
        ):
            return index, region
    LOGGER.warning("No fully interior core exists; profiling the first padded core")
    return 0, regions[0]


def put_template_shards(
    templates: npt.NDArray[np.generic],
    normalization: npt.NDArray[np.float32],
    weights: npt.NDArray[np.float32],
    devices: Sequence[jax.Device],
    batch_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array, int, int]:
    """Build the same padded resident template shards as production scoring."""
    device_count = len(devices)
    templates_per_device = math.ceil(templates.shape[0] / device_count)
    padded_count = math.ceil(templates_per_device / batch_size) * batch_size
    template_shards = []
    normalization_shards = []
    weight_shards = []
    for device_index in range(device_count):
        start = device_index * templates_per_device
        stop = min(start + templates_per_device, templates.shape[0])
        count = max(0, stop - start)
        template_shard = np.zeros(
            (padded_count, *templates.shape[1:]),
            dtype=np.complex64,
        )
        normalization_shard = np.zeros(padded_count, dtype=np.float32)
        weight_shard = np.zeros(padded_count, dtype=np.float32)
        if count:
            template_shard[:count] = np.asarray(
                templates[start:stop], dtype=np.complex64
            )
            normalization_shard[:count] = normalization[start:stop]
            weight_shard[:count] = weights[start:stop]
        template_shards.append(template_shard)
        normalization_shards.append(normalization_shard)
        weight_shards.append(weight_shard)
    device_templates = jax.device_put_sharded(template_shards, devices)
    device_normalization = jax.device_put_sharded(normalization_shards, devices)
    device_weights = jax.device_put_sharded(weight_shards, devices)
    return (
        device_templates,
        device_normalization,
        device_weights,
        templates_per_device,
        padded_count,
    )


def compile_distributed_score(
    devices: Sequence[jax.Device],
    *,
    core_shape: tuple[int, int, int],
    source_shape: tuple[int, int, int],
    whitening_radius: int,
    template_radius: int,
    candidate_capacity: int,
    batch_size: int,
    fft_shape: tuple[int, int, int],
) -> Any:
    """Construct the production template-sharded pmap scoring callable."""
    configured = partial(
        score_template_shards_and_extract_candidates,
        core_shape=core_shape,
        source_shape=source_shape,
        whitening_radius=whitening_radius,
        template_radius=template_radius,
        candidate_capacity=candidate_capacity,
        template_batch_size=batch_size,
        fft_shape=fft_shape,
    )
    return jax.pmap(
        configured,
        axis_name=TEMPLATE_AXIS_NAME,
        in_axes=(None, None, None, 0, 0, 0, None),
        devices=devices,
    )


def synchronize_profile_output(output: Any) -> tuple[int, float]:
    """Match production's device-zero candidate transfer and synchronization."""
    coordinates, scores, counts = (
        np.asarray(output_leaf[0]) for output_leaf in output
    )
    del coordinates
    candidate_count = int(counts)
    finite_scores = scores[np.isfinite(scores)]
    maximum_score = (
        float(np.max(finite_scores)) if finite_scores.size else float("-inf")
    )
    return candidate_count, maximum_score


def execute_iteration(
    distributed_score: Any,
    loaded_subvolume: npt.NDArray[np.float32],
    region_start: npt.NDArray[np.int32],
    whitening_filter: npt.NDArray[np.float32],
    device_templates: jax.Array,
    device_normalization: jax.Array,
    device_weights: jax.Array,
    score_offset: np.float32,
) -> tuple[float, int, float]:
    """Execute and synchronize one exact distributed score invocation."""
    started = time.perf_counter()
    output = distributed_score(
        loaded_subvolume,
        region_start,
        whitening_filter,
        device_templates,
        device_normalization,
        device_weights,
        score_offset,
    )
    candidate_count, maximum_score = synchronize_profile_output(output)
    return time.perf_counter() - started, candidate_count, maximum_score


def main() -> None:
    """Warm up and profile the current checkpointed scoring implementation."""
    args = parse_args()
    configure_logging()
    require_positive("warmup-iterations", args.warmup_iterations)
    require_positive("profile-iterations", args.profile_iterations)
    require_positive("candidate-capacity", args.candidate_capacity)
    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()

    manifest_path = required_checkpoint(results_dir, "00_manifest.json")
    whitening_path = required_checkpoint(results_dir, "03_whitening_filter.npy")
    template_path = required_checkpoint(results_dir, "06b_block_qr_templates.npy")
    score_model_path = required_checkpoint(
        results_dir, "06b_block_qr_score_model.npz"
    )
    manifest = json.loads(manifest_path.read_text())
    input_path = (
        Path(manifest["input"]) if args.input is None else args.input
    ).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(
            f"tomogram is unavailable at {input_path}; pass --input with its path"
        )

    devices = select_devices(args.device_count, allow_cpu=args.allow_cpu)
    core_shape = checkpoint_core_shape(manifest, args.core_patch_shape)
    whitening_filter = np.load(whitening_path, allow_pickle=False)
    if whitening_filter.dtype != np.float32 or whitening_filter.ndim != 3:
        raise ValueError(
            "whitening checkpoint must be a three-dimensional float32 array"
        )
    whitening_radius = spatial_filter_radius(whitening_filter)
    templates = np.load(template_path, mmap_mode="r", allow_pickle=False)
    if templates.ndim != 4 or templates.dtype != np.complex64:
        raise ValueError("block-QR templates must have shape (T,z,y,x) and complex64")
    if len(set(templates.shape[1:])) != 1 or templates.shape[1] % 2 == 0:
        raise ValueError("score templates must have an odd cubic spatial shape")
    template_radius = templates.shape[1] // 2

    with np.load(score_model_path, allow_pickle=False) as model:
        required_fields = {
            "template_normalization",
            "score_weights",
            "score_offset",
        }
        missing_fields = required_fields.difference(model.files)
        if missing_fields:
            raise ValueError(f"score model is missing fields: {sorted(missing_fields)}")
        normalization = np.asarray(
            model["template_normalization"], dtype=np.float32
        )
        weights = np.asarray(model["score_weights"], dtype=np.float32)
        score_offset = np.float32(model["score_offset"].item())
        method = str(model["method"].item()) if "method" in model else "unknown"
    if normalization.shape != (templates.shape[0],):
        raise ValueError("template normalization does not match template count")
    if weights.shape != (templates.shape[0],):
        raise ValueError("score weights do not match template count")

    templates_per_device = math.ceil(templates.shape[0] / len(devices))
    batch_size = resolve_batch_size(
        args.batch_size, results_dir, templates_per_device
    )
    total_halo = whitening_radius + template_radius + 1
    loaded_shape = tuple(size + 2 * total_halo for size in core_shape)
    fft_shape = plan_cufft_fft_shape(loaded_shape)
    candidate_capacity = min(args.candidate_capacity, int(np.prod(core_shape)))

    LOGGER.info("JAX %s | devices=%d", jax.__version__, len(devices))
    for index, device in enumerate(devices):
        LOGGER.info("Device %d: %s", index, device.device_kind)
    LOGGER.info("Input: %s", input_path)
    LOGGER.info(
        "Geometry: core=%s | halo=%d | loaded=%s | FFT=%s | template=%s",
        core_shape,
        total_halo,
        loaded_shape,
        fft_shape,
        templates.shape[1:],
    )
    LOGGER.info(
        "Score model: method=%s | templates=%d | templates/device=%d | "
        "batch=%d | batches/device=%d",
        method,
        templates.shape[0],
        templates_per_device,
        batch_size,
        math.ceil(templates_per_device / batch_size),
    )
    if method != "block_qr_ell_m_v2_fourier_normalized":
        LOGGER.warning(
            "Profiling legacy score-model method %s. Performance geometry is "
            "representative, but these scores are not scientifically current.",
            method,
        )
    if args.dry_run:
        LOGGER.info("Dry run complete; no template shards or FFTs were allocated")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    nvtx = NvtxLibrary(require_active=args.require_nvtx)
    with MrcVolumeSource(input_path) as source:
        if tuple(source.shape) != tuple(manifest["volume_shape_zyx"]):
            raise ValueError(
                f"tomogram shape {source.shape} does not match manifest "
                f"{tuple(manifest['volume_shape_zyx'])}"
            )
        processor = MultiGPUSubvolumeProcessor(
            source,
            domain_shape=source.shape,
            core_shape=core_shape,
            devices=devices,
        )
        region_index, region = select_region(
            processor, total_halo, args.region_index
        )
        LOGGER.info(
            "Profile region %d/%d: start=%s stop=%s",
            region_index,
            processor.subvolume_count,
            region.start,
            region.stop,
        )
        loaded_subvolume = np.asarray(
            processor.load_region(region, total_halo), dtype=np.float32
        )
        expected_loaded_shape = processor.loaded_shape(total_halo)
        if loaded_subvolume.shape != expected_loaded_shape:
            raise RuntimeError(
                f"loaded shape {loaded_subvolume.shape} != {expected_loaded_shape}"
            )

        LOGGER.info("Building and transferring resident template shards")
        (
            device_templates,
            device_normalization,
            device_weights,
            templates_per_device,
            padded_templates_per_device,
        ) = put_template_shards(
            templates,
            normalization,
            weights,
            devices,
            batch_size,
        )
        del templates
        distributed_score = compile_distributed_score(
            devices,
            core_shape=core_shape,
            source_shape=source.shape,
            whitening_radius=whitening_radius,
            template_radius=template_radius,
            candidate_capacity=candidate_capacity,
            batch_size=batch_size,
            fft_shape=fft_shape,
        )
        region_start = np.asarray(region.start, dtype=np.int32)

        warmup_times = []
        LOGGER.info(
            "Warm-up: %d iteration(s), including JIT and cuFFT plan creation",
            args.warmup_iterations,
        )
        for iteration in range(args.warmup_iterations):
            with nvtx.range(f"empiar_scoring_warmup_{iteration}"):
                elapsed, count, maximum = execute_iteration(
                    distributed_score,
                    loaded_subvolume,
                    region_start,
                    whitening_filter,
                    device_templates,
                    device_normalization,
                    device_weights,
                    score_offset,
                )
            warmup_times.append(elapsed)
            LOGGER.info(
                "Warm-up %d/%d: %.3f s | candidates=%d | max-score=%.8g",
                iteration + 1,
                args.warmup_iterations,
                elapsed,
                count,
                maximum,
            )

        memory_before = memory_stats(devices)
        memory_profile_path = output_dir / "device-memory-after-warmup.prof"
        if not args.skip_memory_profile:
            jax.profiler.save_device_memory_profile(memory_profile_path)
            LOGGER.info("JAX device-memory profile: %s", memory_profile_path)

        jax_trace_dir = output_dir / "jax-trace"
        if args.jax_trace:
            jax_trace_dir.mkdir(parents=True, exist_ok=True)
            LOGGER.info("Starting JAX/Perfetto trace: %s", jax_trace_dir)
            jax.profiler.start_trace(
                jax_trace_dir,
                create_perfetto_link=False,
                create_perfetto_trace=True,
            )

        profile_times = []
        candidate_counts = []
        maximum_scores = []
        try:
            LOGGER.info(
                "PROFILE START: range=%s | iterations=%d",
                NVTX_CAPTURE_RANGE,
                args.profile_iterations,
            )
            with nvtx.range(NVTX_CAPTURE_RANGE):
                for iteration in range(args.profile_iterations):
                    with nvtx.range(f"empiar_scoring_iteration_{iteration}"):
                        with jax.profiler.TraceAnnotation(
                            "empiar_scoring_iteration", iteration=iteration
                        ):
                            elapsed, count, maximum = execute_iteration(
                                distributed_score,
                                loaded_subvolume,
                                region_start,
                                whitening_filter,
                                device_templates,
                                device_normalization,
                                device_weights,
                                score_offset,
                            )
                    profile_times.append(elapsed)
                    candidate_counts.append(count)
                    maximum_scores.append(maximum)
                    LOGGER.info(
                        "Profile %d/%d: %.3f s | candidates=%d | max-score=%.8g",
                        iteration + 1,
                        args.profile_iterations,
                        elapsed,
                        count,
                        maximum,
                    )
            LOGGER.info("PROFILE END: range=%s", NVTX_CAPTURE_RANGE)
        finally:
            if args.jax_trace:
                jax.profiler.stop_trace()
                LOGGER.info("JAX/Perfetto trace complete")

        memory_after = memory_stats(devices)

    summary = {
        "jax_version": jax.__version__,
        "backend": devices[0].platform,
        "devices": [device.device_kind for device in devices],
        "input": input_path,
        "results_dir": results_dir,
        "score_model_method": method,
        "core_shape": core_shape,
        "loaded_shape": loaded_subvolume.shape,
        "fft_shape": fft_shape,
        "whitening_radius": whitening_radius,
        "template_radius": template_radius,
        "total_halo": total_halo,
        "template_count": normalization.size,
        "templates_per_device": templates_per_device,
        "padded_templates_per_device": padded_templates_per_device,
        "template_batch_size": batch_size,
        "batches_per_device": padded_templates_per_device // batch_size,
        "candidate_capacity": candidate_capacity,
        "region_index": region_index,
        "region_start": region.start,
        "region_stop": region.stop,
        "warmup_seconds": warmup_times,
        "profile_seconds": profile_times,
        "profile_mean_seconds": float(np.mean(profile_times)),
        "profile_median_seconds": float(np.median(profile_times)),
        "profile_min_seconds": float(np.min(profile_times)),
        "candidate_counts": candidate_counts,
        "maximum_scores": maximum_scores,
        "memory_before_profile": memory_before,
        "memory_after_profile": memory_after,
        "nvtx_capture_range": NVTX_CAPTURE_RANGE,
        "jax_trace_enabled": args.jax_trace,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(json_compatible(summary), indent=2) + "\n")
    LOGGER.info("Profile summary: %s", summary_path)
    LOGGER.info(
        "Steady-state scoring: mean=%.3f s | median=%.3f s | min=%.3f s",
        summary["profile_mean_seconds"],
        summary["profile_median_seconds"],
        summary["profile_min_seconds"],
    )


if __name__ == "__main__":
    main()
