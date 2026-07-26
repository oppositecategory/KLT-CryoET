"""Out-of-core, single-host multi-device radial PSD extraction."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal, Protocol

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from tqdm import tqdm

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


class MultiGPUSubvolumeProcessor:
    """Map subvolume functions across fixed rounds of local GPUs.

    The source, owned core geometry, and devices are shared by every operation.
    Each :meth:`map` call supplies its own halo because neighborhood support is
    a property of the operation rather than the source volume.
    """

    def __init__(
        self,
        source: VolumeSource,
        domain_shape: tuple[int, int, int],
        core_shape: tuple[int, int, int],
        *,
        devices: Sequence[jax.Device] | None = None,
        boundary_mode: PaddingMode = "constant",
        dtype: npt.DTypeLike = np.float32,
    ) -> None:
        """Initialize fixed loading geometry and leading-axis sharding."""
        selected_devices = tuple(jax.devices() if devices is None else devices)
        if not selected_devices:
            raise ValueError("at least one JAX device is required")
        if len({device.platform for device in selected_devices}) != 1:
            raise ValueError("all selected devices must use the same platform")
        if len(domain_shape) != 3 or any(size < 1 for size in domain_shape):
            raise ValueError("domain_shape must contain three positive values")
        if any(
            domain_size > source_size
            for domain_size, source_size in zip(
                domain_shape,
                source.shape,
                strict=True,
            )
        ):
            raise ValueError("domain_shape must be contained by the source")
        if len(core_shape) != 3 or any(size < 1 for size in core_shape):
            raise ValueError("core_shape must contain three positive values")
        if boundary_mode not in {"constant", "edge", "reflect", "wrap"}:
            raise ValueError("unsupported boundary_mode")

        self.source = source
        self.domain_shape = tuple(int(size) for size in domain_shape)
        self.core_shape = tuple(int(size) for size in core_shape)
        self.devices = selected_devices
        self.boundary_mode = boundary_mode
        self.dtype = np.dtype(dtype)

        self._mesh = Mesh(
            np.asarray(self.devices),
            axis_names=("device",),
        )
        self._input_sharding = NamedSharding(
            self._mesh,
            PartitionSpec("device", None, None, None),
        )
        self._region_sharding = NamedSharding(
            self._mesh,
            PartitionSpec("device", None),
        )
        self._replicated_sharding = NamedSharding(
            self._mesh,
            PartitionSpec(),
        )

    def loaded_shape(self, halo: int = 0) -> tuple[int, int, int]:
        """Return the fixed per-device input shape for an operation halo."""
        if halo < 0:
            raise ValueError("halo must be nonnegative")
        return tuple(size + 2 * halo for size in self.core_shape)

    def host_buffer_bytes(self, halo: int = 0) -> int:
        """Return the host staging-buffer size for an operation halo."""
        return (
            len(self.devices)
            * int(np.prod(self.loaded_shape(halo)))
            * self.dtype.itemsize
        )

    @property
    def subvolume_grid_shape(self) -> tuple[int, int, int]:
        """Return the number of core regions along each domain axis."""
        return tuple(
            (domain_size + core_size - 1) // core_size
            for domain_size, core_size in zip(
                self.domain_shape,
                self.core_shape,
                strict=True,
            )
        )

    @property
    def round_count(self) -> int:
        """Return the number of fixed multi-device execution rounds."""
        region_count = int(np.prod(self.subvolume_grid_shape))
        return (region_count + len(self.devices) - 1) // len(self.devices)

    def map(
        self,
        subvolume_function: Callable[..., Any],
        *function_arguments: npt.ArrayLike | jax.Array,
        halo: int = 0,
        pass_region_starts: bool = False,
        static_kwargs: Mapping[str, Any] | None = None,
        description: str = "Subvolume processing",
    ) -> Iterator[tuple[tuple[SpatialRegion | None, ...], Any]]:
        """Yield host outputs from a function applied independently per device.

        ``halo`` controls only the source region loaded for this operation.
        ``pass_region_starts`` inserts each core's global ``(z, y, x)`` start
        as the second argument to ``subvolume_function``.
        ``static_kwargs`` are bound with :func:`functools.partial` before
        vectorization. Positional ``function_arguments`` remain runtime arrays
        and are replicated across devices, allowing same-shaped values to
        change without changing the compiled geometry.
        """
        loaded_shape = self.loaded_shape(halo)
        host_subvolumes = np.empty(
            (len(self.devices), *loaded_shape),
            dtype=self.dtype,
        )
        configured_function = partial(
            subvolume_function,
            **({} if static_kwargs is None else dict(static_kwargs)),
        )
        mapped_axes = (
            (0, 0, *(None for _ in function_arguments))
            if pass_region_starts
            else (0, *(None for _ in function_arguments))
        )
        distributed_function = jax.vmap(configured_function, in_axes=mapped_axes)
        device_arguments = tuple(
            jax.device_put(argument, self._replicated_sharding)
            for argument in function_arguments
        )
        abstract_input = jax.ShapeDtypeStruct(
            host_subvolumes.shape,
            host_subvolumes.dtype,
        )
        abstract_arguments = tuple(
            jax.ShapeDtypeStruct(argument.shape, argument.dtype)
            for argument in device_arguments
        )
        region_starts = np.empty((len(self.devices), 3), dtype=np.int32)
        abstract_regions = jax.ShapeDtypeStruct(
            region_starts.shape,
            region_starts.dtype,
        )
        abstract_inputs = (
            (abstract_input, abstract_regions, *abstract_arguments)
            if pass_region_starts
            else (abstract_input, *abstract_arguments)
        )
        abstract_output = jax.eval_shape(
            distributed_function,
            *abstract_inputs,
        )
        output_shardings = jax.tree_util.tree_map(
            self._output_sharding,
            abstract_output,
        )
        input_shardings = (
            (
                self._input_sharding,
                self._region_sharding,
                *(self._replicated_sharding for _ in device_arguments),
            )
            if pass_region_starts
            else (
                self._input_sharding,
                *(self._replicated_sharding for _ in device_arguments),
            )
        )
        compiled_function = jax.jit(
            distributed_function,
            in_shardings=input_shardings,
            out_shardings=output_shardings,
        )

        for regions in tqdm(
            self._region_rounds(),
            total=self.round_count,
            desc=description,
            unit="round",
        ):
            self._fill_round(host_subvolumes, regions, halo)
            device_subvolumes = jax.device_put(
                host_subvolumes,
                self._input_sharding,
            )
            region_starts.fill(-1)
            for slot, region in enumerate(regions):
                if region is not None:
                    region_starts[slot] = region.start
            compiled_arguments = (
                (
                    device_subvolumes,
                    jax.device_put(region_starts, self._region_sharding),
                    *device_arguments,
                )
                if pass_region_starts
                else (device_subvolumes, *device_arguments)
            )
            device_output = compiled_function(*compiled_arguments)
            host_output = jax.tree_util.tree_map(np.asarray, device_output)
            yield regions, host_output

    def _output_sharding(
        self,
        output: jax.ShapeDtypeStruct,
    ) -> NamedSharding:
        """Shard one vmapped output leaf along its leading device axis."""
        if output.ndim < 1 or output.shape[0] != len(self.devices):
            raise ValueError(
                "subvolume_function outputs must retain the vmapped device axis"
            )
        return NamedSharding(
            self._mesh,
            PartitionSpec(
                "device",
                *(None for _ in range(output.ndim - 1)),
            ),
        )

    def _regions(self) -> Iterator[SpatialRegion]:
        """Yield valid, non-overlapping core regions in traversal order."""
        step_z, step_y, step_x = self.core_shape
        domain_z, domain_y, domain_x = self.domain_shape
        for start_z in range(0, domain_z, step_z):
            for start_y in range(0, domain_y, step_y):
                for start_x in range(0, domain_x, step_x):
                    start = (start_z, start_y, start_x)
                    stop = tuple(
                        min(axis_start + core_size, domain_size)
                        for axis_start, core_size, domain_size in zip(
                            start,
                            self.core_shape,
                            self.domain_shape,
                            strict=True,
                        )
                    )
                    yield SpatialRegion(start, stop)

    def _region_rounds(
        self,
    ) -> Iterator[tuple[SpatialRegion | None, ...]]:
        """Group regions by device count and pad the final round with ``None``."""
        round_regions: list[SpatialRegion | None] = []
        for region in self._regions():
            round_regions.append(region)
            if len(round_regions) == len(self.devices):
                yield tuple(round_regions)
                round_regions = []
        if round_regions:
            round_regions.extend(
                None for _ in range(len(self.devices) - len(round_regions))
            )
            yield tuple(round_regions)

    def _fill_round(
        self,
        host_subvolumes: npt.NDArray[np.generic],
        regions: Sequence[SpatialRegion | None],
        halo: int,
    ) -> None:
        """Fill the reusable host buffer for one execution round."""
        host_subvolumes.fill(0)
        for slot, region in enumerate(regions):
            if region is not None:
                host_subvolumes[slot] = self._load_region(region, halo)

    def _load_region(
        self,
        region: SpatialRegion,
        halo: int,
    ) -> npt.NDArray[np.generic]:
        """Load one fixed core plus halo and pad outside the source."""
        loaded_shape = self.loaded_shape(halo)
        requested_start = tuple(
            start - halo for start in region.start
        )
        requested_stop = tuple(
            start + loaded
            for start, loaded in zip(
                requested_start,
                loaded_shape,
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
            dtype=self.dtype,
        )
        padding = tuple(
            (
                clipped_start[axis] - requested_start[axis],
                requested_stop[axis] - clipped_stop[axis],
            )
            for axis in range(3)
        )
        if any(before or after for before, after in padding):
            if self.boundary_mode == "constant":
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
                    mode=self.boundary_mode,
                )
        if values.shape != loaded_shape:
            raise RuntimeError(
                "loaded subvolume does not match the configured fixed shape"
            )
        return values


def extract_patch_rpsds(
    loaded_subvolume: jax.Array,
    shell_ids: jax.Array,
    shell_counts: jax.Array,
    *,
    patch_size: int,
    core_patch_shape: tuple[int, int, int],
    halo: int,
    patches_per_microbatch: int,
    max_distance: int,
) -> tuple[jax.Array, jax.Array]:
    """Extract patch RPSDs and variances from one fixed loaded subvolume."""
    core_shape = tuple(size * patch_size for size in core_patch_shape)
    core = jax.lax.dynamic_slice(
        loaded_subvolume,
        (halo, halo, halo),
        core_shape,
    )
    blocks_z, blocks_y, blocks_x = core_patch_shape
    patches = core.reshape(
        blocks_z,
        patch_size,
        blocks_y,
        patch_size,
        blocks_x,
        patch_size,
    ).transpose(0, 2, 4, 1, 3, 5)
    patch_count = int(np.prod(core_patch_shape))
    patches = patches.reshape(
        patch_count,
        patch_size,
        patch_size,
        patch_size,
    )

    padded_patch_count = (
        (patch_count + patches_per_microbatch - 1)
        // patches_per_microbatch
        * patches_per_microbatch
    )
    patches = jnp.pad(
        patches,
        (
            (0, padded_patch_count - patch_count),
            (0, 0),
            (0, 0),
            (0, 0),
        ),
    )
    microbatches = patches.reshape(
        -1,
        patches_per_microbatch,
        patch_size,
        patch_size,
        patch_size,
    )

    extract_microbatch = jax.vmap(
        partial(
            estimate_patch_rpsd,
            shell_ids=shell_ids,
            shell_counts=shell_counts,
            max_distance=max_distance,
        )
    )
    rpsds, variances = jax.lax.map(
        extract_microbatch,
        microbatches,
    )
    return (
        rpsds.reshape(padded_patch_count, shell_counts.shape[0])[:patch_count],
        variances.reshape(padded_patch_count)[:patch_count],
    )


def apply_finite_spatial_filter(
    loaded_subvolume: jax.Array,
    spatial_filter: jax.Array,
) -> jax.Array:
    """Apply a centered finite filter by circular FFT convolution.

    Callers must retain outputs at least one filter radius away from every
    loaded boundary. On that interior, circular convolution is identical to
    linear convolution and implements the overlap-save method.
    """
    filter_shape = spatial_filter.shape
    filter_radius = tuple((size - 1) // 2 for size in filter_shape)
    padded_filter = jnp.zeros(
        loaded_subvolume.shape,
        dtype=jnp.result_type(loaded_subvolume, spatial_filter),
    )
    padded_filter = padded_filter.at[
        : filter_shape[0],
        : filter_shape[1],
        : filter_shape[2],
    ].set(spatial_filter)
    padded_filter = jnp.roll(
        padded_filter,
        shift=tuple(-radius for radius in filter_radius),
        axis=(0, 1, 2),
    )
    filtered = jnp.fft.irfftn(
        jnp.fft.rfftn(loaded_subvolume)
        * jnp.fft.rfftn(padded_filter),
        s=loaded_subvolume.shape,
    )
    return filtered.real


def whiten_and_extract_patch_rpsds(
    loaded_subvolume: jax.Array,
    whitening_filter: jax.Array,
    shell_ids: jax.Array,
    shell_counts: jax.Array,
    *,
    patch_size: int,
    core_patch_shape: tuple[int, int, int],
    halo: int,
    patches_per_microbatch: int,
    max_distance: int,
) -> tuple[jax.Array, jax.Array]:
    """Whiten one haloed subvolume and extract RPSDs from its valid core."""
    whitened_subvolume = apply_finite_spatial_filter(
        loaded_subvolume,
        whitening_filter,
    )
    return extract_patch_rpsds(
        whitened_subvolume,
        shell_ids,
        shell_counts,
        patch_size=patch_size,
        core_patch_shape=core_patch_shape,
        halo=halo,
        patches_per_microbatch=patches_per_microbatch,
        max_distance=max_distance,
    )


def spatial_filter_radius(
    spatial_filter: npt.ArrayLike,
) -> int:
    """Validate a finite cubic filter and return its integer radius."""
    filter_array = np.asarray(spatial_filter)
    if filter_array.ndim != 3 or len(set(filter_array.shape)) != 1:
        raise ValueError("spatial_filter must be a cubic three-dimensional array")
    filter_size = filter_array.shape[0]
    if filter_size < 1 or filter_size % 2 == 0:
        raise ValueError("spatial_filter side length must be a positive odd value")
    if not np.all(np.isfinite(filter_array)):
        raise ValueError("spatial_filter must contain only finite values")
    return (filter_size - 1) // 2


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


def extract_streamed_rpsds(
    processor: MultiGPUSubvolumeProcessor,
    patch_size: int,
    *,
    patches_per_microbatch: int = 8,
    max_acf_distance_fraction: float = 0.3,
    spatial_filter: npt.ArrayLike | None = None,
) -> RpsdExtractionResult:
    """Extract all patch RPSDs using a reusable multi-GPU processor."""
    if patch_size < 3 or patch_size % 2 == 0:
        raise ValueError("patch_size must be an odd integer at least 3")
    if patches_per_microbatch < 1:
        raise ValueError("patches_per_microbatch must be positive")
    if not 0 < max_acf_distance_fraction < 1:
        raise ValueError("max_acf_distance_fraction must lie in (0, 1)")
    if any(size % patch_size for size in processor.core_shape):
        raise ValueError("processor core_shape must align to complete patches")
    core_patch_shape = tuple(
        size // patch_size for size in processor.core_shape
    )
    patch_grid_shape = tuple(
        size // patch_size for size in processor.domain_shape
    )
    radial_points, shell_ids, shell_counts = (
        generate_uniform_radial_sampling_points(2 * patch_size - 1, np.pi)
    )
    rpsd_grid = np.empty(
        (*patch_grid_shape, radial_points.size),
        dtype=np.float32,
    )
    variance_grid = np.empty(patch_grid_shape, dtype=np.float32)
    max_distance = max(
        1,
        int(np.floor(max_acf_distance_fraction * patch_size)),
    )
    static_kwargs = {
        "patch_size": patch_size,
        "core_patch_shape": core_patch_shape,
        "patches_per_microbatch": patches_per_microbatch,
        "max_distance": max_distance,
    }

    if spatial_filter is None:
        halo = 0
        subvolume_function = extract_patch_rpsds
        function_arguments = (shell_ids, shell_counts)
        description = "RPSD extraction"
    else:
        spatial_filter = np.asarray(spatial_filter, dtype=np.float32)
        halo = spatial_filter_radius(spatial_filter)
        subvolume_function = whiten_and_extract_patch_rpsds
        function_arguments = (spatial_filter, shell_ids, shell_counts)
        description = "Whitened RPSD extraction"
    static_kwargs["halo"] = halo

    outputs = processor.map(
        subvolume_function,
        *function_arguments,
        halo=halo,
        static_kwargs=static_kwargs,
        description=description,
    )
    for regions, (host_rpsds, host_variances) in outputs:
        for slot, region in enumerate(regions):
            if region is None:
                continue
            patch_start = tuple(start // patch_size for start in region.start)
            valid = tuple(
                (stop - start) // patch_size
                for start, stop in zip(
                    region.start,
                    region.stop,
                    strict=True,
                )
            )
            destination = tuple(
                slice(start, start + size)
                for start, size in zip(patch_start, valid, strict=True)
            )
            local = tuple(slice(0, size) for size in valid)
            local_rpsds = host_rpsds[slot].reshape(
                *core_patch_shape,
                radial_points.size,
            )
            local_variances = host_variances[slot].reshape(core_patch_shape)
            rpsd_grid[destination] = local_rpsds[local]
            variance_grid[destination] = local_variances[local]

    return RpsdExtractionResult(
        rpsds=rpsd_grid.reshape(-1, radial_points.size),
        variances=variance_grid.ravel(),
        radial_points=radial_points.copy(),
        patch_grid_shape=patch_grid_shape,
        patch_size=patch_size,
    )


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
