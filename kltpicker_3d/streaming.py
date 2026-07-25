"""Out-of-core, single-host multi-device radial PSD extraction."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from kltpicker_3d.spectral_estimation import (
    estimate_isotropic_powerspectrum_tensor,
)
from kltpicker_3d.utils import (
    generate_uniform_radial_sampling_points,
    radial_average_jax,
)

PaddingMode = Literal["constant", "edge", "reflect", "wrap"]


def default_psd_patch_size(
    particle_diameter: float,
    mgscale: float,
) -> int:
    """Return the current detector's default odd PSD patch side."""
    if particle_diameter <= 0:
        raise ValueError("particle_diameter must be positive")
    if mgscale <= 0:
        raise ValueError("mgscale must be positive")
    size = int(np.floor(0.8 * particle_diameter * mgscale))
    size = size if size % 2 else size - 1
    if size < 3:
        raise ValueError(
            "particle_diameter * mgscale must yield a PSD patch of at least 3"
        )
    return size


def estimate_patch_rpsd(
    patch: jax.Array | npt.ArrayLike,
    shell_ids: jax.Array | npt.ArrayLike,
    shell_counts: jax.Array | npt.ArrayLike,
    max_distance: int,
) -> tuple[jax.Array, jax.Array]:
    """Return one mean-centered patch's radial PSD and spatial variance."""
    centered = jnp.asarray(patch) - jnp.mean(patch)
    spectrum = estimate_isotropic_powerspectrum_tensor(
        centered,
        max_distance,
    )
    rpsd = radial_average_jax(
        spectrum,
        shell_ids,
        shell_counts,
        shell_counts.shape[0],
    )
    return rpsd, jnp.var(centered)


@dataclass(frozen=True)
class SpatialRegion:
    """Half-open ``(z, y, x)`` region."""

    start: tuple[int, int, int]
    stop: tuple[int, int, int]


class VolumeSource(Protocol):
    """Random-access source for a disk- or memory-backed volume."""

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return the volume shape in ``(z, y, x)`` order."""

    def read(self, region: SpatialRegion) -> npt.NDArray[np.generic]:
        """Read an in-bounds region as a NumPy array."""


class ArrayVolumeSource:
    """Expose a NumPy-compatible three-dimensional array as a volume source."""

    def __init__(self, volume: npt.ArrayLike) -> None:
        """Initialize the source without copying an existing NumPy array."""
        array = np.asanyarray(volume)
        if array.ndim != 3:
            raise ValueError("volume must be three-dimensional")
        self._volume = array

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return the volume shape."""
        return tuple(int(size) for size in self._volume.shape)

    def read(self, region: SpatialRegion) -> npt.NDArray[np.generic]:
        """Read an in-bounds region."""
        _validate_region(region, self.shape)
        slices = tuple(
            slice(axis_start, axis_stop)
            for axis_start, axis_stop in zip(
                region.start,
                region.stop,
                strict=True,
            )
        )
        return np.asarray(self._volume[slices])


class MrcVolumeSource:
    """Memory-mapped MRC volume source.

    ``mrcfile`` is imported lazily so array-backed streaming does not require
    the optional file-format dependency.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        permissive: bool = True,
    ) -> None:
        """Open an MRC file read-only using memory mapping."""
        try:
            import mrcfile
        except ImportError as error:
            raise ImportError(
                "MrcVolumeSource requires the 'mrcfile' package"
            ) from error
        self._mrc = mrcfile.mmap(
            Path(path),
            mode="r",
            permissive=permissive,
        )
        if self._mrc.data.ndim != 3:
            self._mrc.close()
            raise ValueError("MRC volume must be three-dimensional")

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return the volume shape."""
        return tuple(int(size) for size in self._mrc.data.shape)

    def read(self, region: SpatialRegion) -> npt.NDArray[np.generic]:
        """Read an in-bounds region from the memory map."""
        _validate_region(region, self.shape)
        slices = tuple(
            slice(axis_start, axis_stop)
            for axis_start, axis_stop in zip(
                region.start,
                region.stop,
                strict=True,
            )
        )
        return np.asarray(self._mrc.data[slices])

    def close(self) -> None:
        """Close the underlying MRC memory map."""
        self._mrc.close()

    def __enter__(self) -> MrcVolumeSource:
        """Return this open source."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close the source on context-manager exit."""
        self.close()


@dataclass(frozen=True)
class _SubvolumeTask:
    """One fixed-shape device slot and its valid global patch ownership."""

    patch_start: tuple[int, int, int]
    valid_patch_shape: tuple[int, int, int]
    is_dummy: bool = False


@dataclass(frozen=True)
class StreamingRpsdConfig:
    """Static execution geometry for streamed radial PSD extraction."""

    patch_size: int
    core_patch_shape: tuple[int, int, int]
    patches_per_microbatch: int = 8
    halo: int = 0
    boundary_mode: PaddingMode = "constant"
    max_acf_distance_fraction: float = 0.3

    def __post_init__(self) -> None:
        """Validate fixed compiled geometry."""
        if self.patch_size < 3 or self.patch_size % 2 == 0:
            raise ValueError("patch_size must be an odd integer at least 3")
        if len(self.core_patch_shape) != 3 or any(
            size < 1 for size in self.core_patch_shape
        ):
            raise ValueError("core_patch_shape must contain three positive values")
        if self.patches_per_microbatch < 1:
            raise ValueError("patches_per_microbatch must be positive")
        if self.halo < 0:
            raise ValueError("halo must be nonnegative")
        if self.boundary_mode not in {"constant", "edge", "reflect", "wrap"}:
            raise ValueError("unsupported boundary_mode")
        if not 0 < self.max_acf_distance_fraction < 1:
            raise ValueError("max_acf_distance_fraction must lie in (0, 1)")

    @property
    def core_shape(self) -> tuple[int, int, int]:
        """Return the owned voxel shape, aligned to complete patches."""
        return tuple(size * self.patch_size for size in self.core_patch_shape)

    @property
    def loaded_shape(self) -> tuple[int, int, int]:
        """Return the fixed source-buffer shape including halos."""
        return tuple(size + 2 * self.halo for size in self.core_shape)

    @property
    def patches_per_subvolume(self) -> int:
        """Return the number of allocated patches in every device slot."""
        return int(np.prod(self.core_patch_shape))

    @property
    def padded_patch_count(self) -> int:
        """Return the patch count rounded up to a complete microbatch."""
        batch = self.patches_per_microbatch
        count = self.patches_per_subvolume
        return ((count + batch - 1) // batch) * batch


@dataclass(frozen=True)
class RpsdExtractionResult:
    """Compact host-resident output of streamed patch reduction."""

    rpsds: npt.NDArray[np.float32]
    variances: npt.NDArray[np.float32]
    radial_points: npt.NDArray[np.float64]
    patch_grid_shape: tuple[int, int, int]
    patch_size: int


def estimate_device_memory_limit(
    devices: Sequence[jax.Device],
) -> int | None:
    """Return the smallest allocator limit reported by selected devices.

    JAX backends are not required to expose memory statistics. ``None`` means
    that automatic capacity planning is unavailable and the caller should use
    an explicit ``core_patch_shape``.
    """
    limits = []
    for device in devices:
        statistics = device.memory_stats()
        if not statistics:
            return None
        limit = statistics.get("bytes_limit")
        if limit is None:
            limit = statistics.get("bytes_reservable_limit")
        if limit is None:
            return None
        limits.append(int(limit))
    return min(limits)


def suggest_core_patch_shape(
    patch_size: int,
    device_memory_bytes: int,
    *,
    memory_fraction: float = 0.5,
    resident_volume_copies: int = 8,
    halo: int = 0,
) -> tuple[int, int, int]:
    """Suggest a conservative cubic patch grid from a static memory budget.

    This is an initial geometry estimate, not a proof that an XLA executable
    will fit. FFT workspaces and compiled temporary storage are backend
    dependent, so production runs should validate the returned shape by
    compiling and warming up the complete kernel.
    """
    if patch_size < 1:
        raise ValueError("patch_size must be positive")
    if device_memory_bytes < 1:
        raise ValueError("device_memory_bytes must be positive")
    if not 0 < memory_fraction < 1:
        raise ValueError("memory_fraction must lie in (0, 1)")
    if resident_volume_copies < 1:
        raise ValueError("resident_volume_copies must be positive")
    if halo < 0:
        raise ValueError("halo must be nonnegative")

    usable_voxels = int(
        device_memory_bytes
        * memory_fraction
        / (np.dtype(np.float32).itemsize * resident_volume_copies)
    )
    loaded_side = int(round(usable_voxels ** (1 / 3)))
    while loaded_side**3 > usable_voxels:
        loaded_side -= 1
    core_side = loaded_side - 2 * halo
    patches_per_axis = core_side // patch_size
    if patches_per_axis < 1:
        raise ValueError("device memory budget cannot contain one loaded patch")
    return (patches_per_axis,) * 3


class ShardedRpsdEstimator:
    """Synchronously stream fixed subvolume rounds across local devices."""

    @classmethod
    def from_particle_geometry(
        cls,
        source: VolumeSource,
        particle_diameter: float,
        mgscale: float,
        *,
        devices: Sequence[jax.Device] | None = None,
        core_patch_shape: tuple[int, int, int] | None = None,
        device_memory_bytes: int | None = None,
        memory_fraction: float = 0.5,
        resident_volume_copies: int = 8,
        patches_per_microbatch: int = 8,
        halo: int = 0,
        boundary_mode: PaddingMode = "constant",
    ) -> ShardedRpsdEstimator:
        """Construct an extractor from particle and device geometry.

        If ``core_patch_shape`` is omitted, the method queries the smallest
        selected-device memory limit and chooses a conservative cubic core.
        ``device_memory_bytes`` provides an explicit fallback for backends
        that do not report memory statistics.
        """
        selected_devices = tuple(
            jax.devices() if devices is None else devices
        )
        if not selected_devices:
            raise ValueError("at least one JAX device is required")
        patch_size = default_psd_patch_size(
            particle_diameter,
            mgscale,
        )
        if core_patch_shape is None:
            memory_limit = (
                device_memory_bytes
                if device_memory_bytes is not None
                else estimate_device_memory_limit(selected_devices)
            )
            if memory_limit is None:
                raise ValueError(
                    "device memory is unavailable; provide core_patch_shape "
                    "or device_memory_bytes"
                )
            suggested_shape = suggest_core_patch_shape(
                patch_size,
                memory_limit,
                memory_fraction=memory_fraction,
                resident_volume_copies=resident_volume_copies,
                halo=halo,
            )
            volume_patch_grid = tuple(
                size // patch_size for size in source.shape
            )
            if any(size < 1 for size in volume_patch_grid):
                raise ValueError(
                    "every volume axis must contain at least one patch"
                )
            core_patch_shape = tuple(
                min(suggested, available)
                for suggested, available in zip(
                    suggested_shape,
                    volume_patch_grid,
                    strict=True,
                )
            )

        config = StreamingRpsdConfig(
            patch_size=patch_size,
            core_patch_shape=core_patch_shape,
            patches_per_microbatch=patches_per_microbatch,
            halo=halo,
            boundary_mode=boundary_mode,
        )
        return cls(
            source,
            config,
            devices=selected_devices,
        )

    def __init__(
        self,
        source: VolumeSource,
        config: StreamingRpsdConfig,
        *,
        devices: Sequence[jax.Device] | None = None,
    ) -> None:
        """Initialize fixed geometry, host buffers, and sharded computation."""
        selected_devices = tuple(jax.devices() if devices is None else devices)
        if not selected_devices:
            raise ValueError("at least one JAX device is required")
        if len({device.platform for device in selected_devices}) != 1:
            raise ValueError("all selected devices must use the same platform")

        self.source = source
        self.config = config
        self.devices = selected_devices
        self.patch_grid_shape = tuple(
            size // config.patch_size for size in source.shape
        )
        if any(size < 1 for size in self.patch_grid_shape):
            raise ValueError("every volume axis must contain at least one patch")
        self.uniform_points, shell_ids, counts = (
            generate_uniform_radial_sampling_points(
                2 * config.patch_size - 1,
                np.pi,
            )
        )
        self._shell_ids = jnp.asarray(shell_ids)
        self._counts = jnp.asarray(counts)
        self._host_subvolumes = np.empty(
            (len(selected_devices), *config.loaded_shape),
            dtype=np.float32,
        )

        mesh = Mesh(np.asarray(selected_devices), axis_names=("device",))
        self._input_sharding = NamedSharding(
            mesh,
            PartitionSpec("device", None, None, None),
        )
        self._rpsd_sharding = NamedSharding(
            mesh,
            PartitionSpec("device", None, None),
        )
        self._variance_sharding = NamedSharding(
            mesh,
            PartitionSpec("device", None),
        )
        self._sharded_step = self._build_sharded_step()

    @property
    def host_buffer_bytes(self) -> int:
        """Return bytes held by the reusable input buffer."""
        return self._host_subvolumes.nbytes

    def _build_sharded_step(
        self,
    ) -> Callable[
        [jax.Array],
        tuple[jax.Array, jax.Array],
    ]:
        """Compile the fixed global subvolume-to-RPSD computation."""
        config = self.config
        shell_ids = self._shell_ids
        counts = self._counts
        radial_bin_count = self.uniform_points.size
        max_distance = max(
            1,
            int(
                np.floor(
                    config.max_acf_distance_fraction * config.patch_size
                )
            ),
        )

        def process_patch(
            patch: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            return estimate_patch_rpsd(
                patch,
                shell_ids,
                counts,
                max_distance,
            )

        process_patch_microbatch = jax.vmap(process_patch)

        def process_subvolume(
            loaded: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            halo = config.halo
            core_z, core_y, core_x = config.core_shape
            core = jax.lax.dynamic_slice(
                loaded,
                (halo, halo, halo),
                (core_z, core_y, core_x),
            )
            blocks_z, blocks_y, blocks_x = config.core_patch_shape
            patch_size = config.patch_size
            patches = core.reshape(
                blocks_z,
                patch_size,
                blocks_y,
                patch_size,
                blocks_x,
                patch_size,
            ).transpose(0, 2, 4, 1, 3, 5)
            patches = patches.reshape(
                config.patches_per_subvolume,
                patch_size,
                patch_size,
                patch_size,
            )

            padding = config.padded_patch_count - config.patches_per_subvolume
            patches = jnp.pad(
                patches,
                ((0, padding), (0, 0), (0, 0), (0, 0)),
            )
            microbatches = patches.reshape(
                -1,
                config.patches_per_microbatch,
                patch_size,
                patch_size,
                patch_size,
            )

            rpsds, variances = jax.lax.map(
                process_patch_microbatch,
                microbatches,
            )
            rpsds = rpsds.reshape(
                config.padded_patch_count,
                radial_bin_count,
            )[: config.patches_per_subvolume]
            variances = variances.reshape(
                config.padded_patch_count
            )[: config.patches_per_subvolume]
            return rpsds, variances

        global_step = jax.vmap(process_subvolume)
        return jax.jit(
            global_step,
            in_shardings=(self._input_sharding,),
            out_shardings=(
                self._rpsd_sharding,
                self._variance_sharding,
            ),
        )

    def extract(self) -> RpsdExtractionResult:
        """Run all fixed sharded rounds and return compact host arrays."""
        rpsd_grid = np.empty(
            (*self.patch_grid_shape, self.uniform_points.size),
            dtype=np.float32,
        )
        variance_grid = np.empty(self.patch_grid_shape, dtype=np.float32)

        for tasks in self._task_rounds():
            self._fill_round(tasks)
            device_subvolumes = jax.device_put(
                self._host_subvolumes,
                self._input_sharding,
            )
            device_rpsds, device_variances = self._sharded_step(
                device_subvolumes,
            )
            host_rpsds = np.asarray(device_rpsds)
            host_variances = np.asarray(device_variances)
            for slot, task in enumerate(tasks):
                if task.is_dummy:
                    continue
                valid = task.valid_patch_shape
                destination = tuple(
                    slice(start, start + size)
                    for start, size in zip(
                        task.patch_start,
                        valid,
                        strict=True,
                    )
                )
                local = tuple(slice(0, size) for size in valid)
                local_rpsds = host_rpsds[slot].reshape(
                    *self.config.core_patch_shape,
                    self.uniform_points.size,
                )
                local_variances = host_variances[slot].reshape(
                    self.config.core_patch_shape
                )
                rpsd_grid[destination] = local_rpsds[local]
                variance_grid[destination] = local_variances[local]

        return RpsdExtractionResult(
            rpsds=rpsd_grid.reshape(-1, self.uniform_points.size),
            variances=variance_grid.ravel(),
            radial_points=self.uniform_points.copy(),
            patch_grid_shape=self.patch_grid_shape,
            patch_size=self.config.patch_size,
        )

    def _task_rounds(self) -> Iterator[tuple[_SubvolumeTask, ...]]:
        """Yield fixed rounds of spatial tasks, padding only the last round."""
        step_z, step_y, step_x = self.config.core_patch_shape
        grid_z, grid_y, grid_x = self.patch_grid_shape
        tasks = (
            _SubvolumeTask(
                patch_start=(start_z, start_y, start_x),
                valid_patch_shape=(
                    min(step_z, grid_z - start_z),
                    min(step_y, grid_y - start_y),
                    min(step_x, grid_x - start_x),
                ),
            )
            for start_z in range(0, grid_z, step_z)
            for start_y in range(0, grid_y, step_y)
            for start_x in range(0, grid_x, step_x)
        )

        round_tasks: list[_SubvolumeTask] = []
        for task in tasks:
            round_tasks.append(task)
            if len(round_tasks) == len(self.devices):
                yield tuple(round_tasks)
                round_tasks = []
        if round_tasks:
            round_tasks.extend(
                _SubvolumeTask(
                    patch_start=(0, 0, 0),
                    valid_patch_shape=(0, 0, 0),
                    is_dummy=True,
                )
                for _ in range(len(self.devices) - len(round_tasks))
            )
            yield tuple(round_tasks)

    def _fill_round(self, tasks: Sequence[_SubvolumeTask]) -> None:
        """Fill reusable host buffers for one fixed device round."""
        self._host_subvolumes.fill(0)
        for slot, task in enumerate(tasks):
            if task.is_dummy:
                continue
            self._host_subvolumes[slot] = self._load_task(task)

    def _load_task(
        self,
        task: _SubvolumeTask,
    ) -> npt.NDArray[np.float32]:
        """Load a task and pad it to the fixed haloed device shape."""
        patch_size = self.config.patch_size
        halo = self.config.halo
        core_start = tuple(index * patch_size for index in task.patch_start)
        requested_start = tuple(index - halo for index in core_start)
        requested_stop = tuple(
            start + loaded
            for start, loaded in zip(
                requested_start,
                self.config.loaded_shape,
                strict=True,
            )
        )

        clipped_start = tuple(max(0, start) for start in requested_start)
        clipped_stop = tuple(
            min(stop, size)
            for stop, size in zip(
                requested_stop,
                self.source.shape,
                strict=True,
            )
        )
        values = np.asarray(
            self.source.read(SpatialRegion(clipped_start, clipped_stop)),
            dtype=np.float32,
        )
        padding = tuple(
            (
                clipped_start[axis] - requested_start[axis],
                requested_stop[axis] - clipped_stop[axis],
            )
            for axis in range(3)
        )
        if any(before or after for before, after in padding):
            if self.config.boundary_mode == "constant":
                values = np.pad(
                    values,
                    padding,
                    mode="constant",
                    constant_values=0,
                )
            else:
                values = np.pad(
                    values,
                    padding,
                    mode=self.config.boundary_mode,
                )
        if values.shape != self.config.loaded_shape:
            raise RuntimeError(
                "loaded subvolume does not match the configured fixed shape"
            )
        return values


def _validate_region(
    region: SpatialRegion,
    volume_shape: tuple[int, int, int],
) -> None:
    """Validate a nonempty region contained by a volume."""
    if len(region.start) != 3 or len(region.stop) != 3:
        raise ValueError("region must be three-dimensional")
    for axis_start, axis_stop, size in zip(
        region.start,
        region.stop,
        volume_shape,
        strict=True,
    ):
        if not 0 <= axis_start < axis_stop <= size:
            raise ValueError("region must be nonempty and inside the volume")
