"""Extract patch RPSDs from an EMPIAR-10045 ribosome tomogram."""

from __future__ import annotations

import argparse
import logging
import math
import os
import pickle
import tempfile
import time
from pathlib import Path

import jax
import numpy as np

from kltpicker_3d.streaming import MrcVolumeSource, ShardedRpsdEstimator

DEFAULT_INPUT = Path(
    "/luke_leia_data/yoelsh/datasets/10045/pristine/data/ribosomes/"
    "Tomograms/08/IS002_291013_008.mrc"
)
LOGGER = logging.getLogger("empiar-10045")


def parse_args() -> argparse.Namespace:
    """Parse the experiment configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("empiar-10045-rpsds.pkl"),
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("empiar-10045.log"),
    )
    parser.add_argument(
        "--particle-diameter",
        type=float,
        default=270.0,
        metavar="ANGSTROM",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        metavar="ANGSTROM",
        help="Override the isotropic voxel size in the MRC header.",
    )
    parser.add_argument(
        "--core-patch-shape",
        type=int,
        nargs=3,
        metavar=("Z", "Y", "X"),
        help="Patches per GPU subvolume; default: infer from GPU memory.",
    )
    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=0.3,
        help="Fraction of JAX's per-device allocator limit used for planning.",
    )
    parser.add_argument(
        "--patches-per-microbatch",
        type=int,
        default=1,
        help="Simultaneous patch FFTs per GPU; increase only after a stable run.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU execution for small debugging runs.",
    )
    return parser.parse_args()


def configure_logging(log_file: Path) -> None:
    """Log to both stderr and a persistent file."""
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
    """Return an isotropic voxel size and the original (z, y, x) spacing."""
    try:
        import mrcfile
    except ImportError as error:
        raise RuntimeError(
            "mrcfile is missing; install the project's declared dependencies"
        ) from error

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
        raise ValueError(
            f"invalid MRC voxel spacing {spacing}; pass --voxel-size"
        )
    if not np.allclose(values, values[0], rtol=1e-4):
        raise ValueError(
            f"anisotropic MRC voxel spacing {spacing}; resample the volume first"
        )
    return float(values.mean()), spacing


def local_devices(*, allow_cpu: bool) -> tuple[jax.Device, ...]:
    """Use every local GPU and reject accidental login-node execution."""
    if jax.process_count() != 1:
        raise RuntimeError(
            "ShardedRpsdEstimator is single-host; expected one JAX process, "
            f"found {jax.process_count()}"
        )
    visible = tuple(jax.local_devices())
    gpus = tuple(device for device in visible if device.platform == "gpu")
    if gpus:
        return gpus
    if allow_cpu:
        return visible
    raise RuntimeError(
        "JAX sees no GPU. Run on a GPU node, or pass --allow-cpu only for "
        "a small debugging run."
    )


def memory_limit_gib(device: jax.Device) -> float | None:
    """Read JAX's per-device allocator limit when the backend exposes it."""
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


def log_plan(
    path: Path,
    particle_diameter: float,
    voxel_size: float,
    header_spacing: tuple[float, float, float],
    estimator: ShardedRpsdEstimator,
) -> None:
    """Log the resolved geometry before the expensive first compilation."""
    config = estimator.config
    grid = estimator.patch_grid_shape
    patch_count = math.prod(grid)
    used_shape = tuple(count * config.patch_size for count in grid)
    cropped = tuple(
        size - used
        for size, used in zip(estimator.source.shape, used_shape, strict=True)
    )
    result_mib = (
        patch_count
        * (estimator.uniform_points.size + 1)
        * np.dtype(np.float32).itemsize
        + estimator.uniform_points.nbytes
    ) / 2**20

    LOGGER.info(
        "Input: %s | %.2f GiB | shape(z,y,x)=%s",
        path,
        path.stat().st_size / 2**30,
        estimator.source.shape,
    )
    LOGGER.info(
        "Particle: %.1f A | voxel: %.7f A (header z,y,x=%s)",
        particle_diameter,
        voxel_size,
        tuple(round(value, 7) for value in header_spacing),
    )
    LOGGER.info(
        "Patch: %d^3 voxels (%.1f A) | grid=%s | patches=%d",
        config.patch_size,
        config.patch_size * voxel_size,
        grid,
        patch_count,
    )
    LOGGER.info(
        "RPSD: max ACF distance=%d voxels | radial bins=%d",
        max(
            1,
            math.floor(
                config.max_acf_distance_fraction * config.patch_size
            ),
        ),
        estimator.uniform_points.size,
    )
    LOGGER.info(
        "FFT workload per GPU: batch=%d | padded shape=%s",
        config.patches_per_microbatch,
        (2 * config.patch_size - 1,) * 3,
    )
    if any(cropped):
        LOGGER.warning("Trailing cropped voxels (z,y,x): %s", cropped)
    LOGGER.info(
        "Per GPU: core patches=%s | loaded voxels=%s | microbatch=%d",
        config.core_patch_shape,
        config.loaded_shape,
        config.patches_per_microbatch,
    )
    LOGGER.info(
        "Schedule: subvolume grid=%s | rounds=%d | host buffer=%.2f GiB",
        estimator.subvolume_grid_shape,
        estimator.round_count,
        estimator.host_buffer_bytes / 2**30,
    )
    LOGGER.info(
        "Expected result: RPSDs=(%d, %d), variances=(%d,) | %.1f MiB",
        patch_count,
        estimator.uniform_points.size,
        patch_count,
        result_mib,
    )


def save_result(result: object, output: Path, *, overwrite: bool) -> None:
    """Atomically publish the result so a failed write is never mistaken as valid."""
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"{output} already exists; pass --overwrite to replace it"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main() -> None:
    """Validate configuration, run extraction, and save the host result."""
    args = parse_args()
    configure_logging(args.log_file)
    started = time.perf_counter()

    try:
        if not args.input.is_file():
            raise FileNotFoundError(args.input)
        if args.input.resolve() == args.output.resolve():
            raise ValueError("input and output paths must differ")
        if args.log_file.resolve() == args.output.resolve():
            raise ValueError("log and output paths must differ")
        if args.output.exists() and not args.overwrite and not args.dry_run:
            raise FileExistsError(
                f"{args.output} already exists; pass --overwrite to replace it"
            )

        voxel_size, header_spacing = mrc_voxel_size(
            args.input,
            args.voxel_size,
        )

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
            estimator = ShardedRpsdEstimator.from_particle_geometry(
                source,
                particle_diameter=args.particle_diameter,
                mgscale=1.0 / voxel_size,
                devices=devices,
                core_patch_shape=(
                    None
                    if args.core_patch_shape is None
                    else tuple(args.core_patch_shape)
                ),
                memory_fraction=args.memory_fraction,
                patches_per_microbatch=args.patches_per_microbatch,
            )
            log_plan(
                args.input,
                args.particle_diameter,
                voxel_size,
                header_spacing,
                estimator,
            )
            if args.dry_run:
                LOGGER.info("Dry run complete; extraction was not started.")
                return

            LOGGER.info("Starting sharded RPSD extraction")
            extraction_started = time.perf_counter()
            result = estimator.extract()
            LOGGER.info(
                "Extraction completed in %.1f minutes",
                (time.perf_counter() - extraction_started) / 60,
            )

        LOGGER.info("Writing %s", args.output)
        save_result(result, args.output, overwrite=args.overwrite)
        LOGGER.info(
            "Done in %.1f minutes | output size=%.2f MiB",
            (time.perf_counter() - started) / 60,
            args.output.stat().st_size / 2**20,
        )
    except Exception:
        LOGGER.exception(
            "Experiment failed after %.1f minutes",
            (time.perf_counter() - started) / 60,
        )
        raise


if __name__ == "__main__":
    main()
