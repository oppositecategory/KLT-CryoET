"""Validate EMPIAR template construction and sharded scoring checkpoints.

The diagnostic has four explicit modes so single-device and multi-device JAX
executions can run in fresh processes (and therefore fresh GPU allocators):

``templates``
    Rebuild the radial model and raw templates with the original host
    implementation, then independently reproduce every ``(ell, m)`` QR block
    with NumPy and compare it with the saved distributed JAX result.
``score-single``
    Score one production-shaped haloed crop block-by-block on one device using
    the unfused spatial-convolution reference implementation.
``score-sharded``
    Score the identical saved crop with template sharding, fused FFTs, and
    ``lax.psum``. Both per-block and production contiguous sharding are saved.
``compare``
    Compare the dense checkpoint volumes before local maxima and NMS.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
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
    compute_fused_klt_score_shard,
    compute_klt_score_block,
    extract_score_candidates,
)
from kltpicker_3d.streaming import (
    MrcVolumeSource,
    MultiGPUSubvolumeProcessor,
    SpatialRegion,
)
from kltpicker_3d.tomogram import KLTParticleDetector3D


DEFAULT_RESULTS = REPOSITORY_ROOT / "results/empiar-10045-bandpass-block-qr"
DEFAULT_OUTPUT = DEFAULT_RESULTS / "distributed_validation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("templates", "score-single", "score-sharded", "compare"),
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--region-start",
        type=int,
        nargs=3,
        default=(186, 1488, 1488),
        metavar=("Z", "Y", "X"),
        help="Production core start; the default contains a top false peak.",
    )
    parser.add_argument("--template-batch-size", type=int, default=1)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        return json.load(stream)


def save_json(value: dict[str, Any], path: Path) -> None:
    with path.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def load_pickle(path: Path) -> Any:
    with path.open("rb") as stream:
        return pickle.load(stream)


def block_indices(
    orders: npt.NDArray[np.int64],
    m_values: npt.NDArray[np.int64],
) -> list[tuple[int, int, npt.NDArray[np.int64]]]:
    return [
        (
            order,
            m_value,
            np.flatnonzero((orders == order) & (m_values == m_value)),
        )
        for order, m_value in sorted(set(zip(orders.tolist(), m_values.tolist())))
    ]


def phase_aligned_relative_errors(
    reference: npt.NDArray[np.generic],
    observed: npt.NDArray[np.generic],
) -> npt.NDArray[np.float64]:
    errors = []
    for reference_template, observed_template in zip(reference, observed, strict=True):
        a = np.asarray(reference_template).reshape(-1).astype(np.complex128)
        b = np.asarray(observed_template).reshape(-1).astype(np.complex128)
        inner = np.vdot(a, b)
        phase = 1.0 if inner == 0 else np.conj(inner) / abs(inner)
        errors.append(np.linalg.norm(a - phase * b) / np.linalg.norm(a))
    return np.asarray(errors)


def cpu_qr_score_block(
    templates: npt.NDArray[np.generic],
    eigenvalues: npt.NDArray[np.float64],
    noise_variance: float,
) -> tuple[
    npt.NDArray[np.complex128],
    npt.NDArray[np.float64],
    float,
    npt.NDArray[np.float64],
]:
    width = templates.shape[0]
    matrix = np.asarray(templates, dtype=np.complex128).reshape(width, -1).T
    basis, triangular = np.linalg.qr(matrix, mode="reduced")
    covariance = (triangular * eigenvalues[None, :]) @ triangular.conj().T
    covariance = 0.5 * (covariance + covariance.conj().T)
    signal_values, eigenvectors = np.linalg.eigh(covariance)
    signal_values = np.maximum(signal_values, 0)[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    covariance_values = noise_variance + signal_values
    weights = 1.0 / noise_variance - 1.0 / covariance_values
    offset = float(np.sum(np.log(covariance_values / noise_variance)))
    return (basis @ eigenvectors).T, weights, offset, signal_values


def orthonormalize_rows_from_gram(
    row_basis: npt.NDArray[np.complex128],
) -> npt.NDArray[np.complex128]:
    gram = row_basis.conj() @ row_basis.T
    values, vectors = np.linalg.eigh(0.5 * (gram + gram.conj().T))
    inverse_sqrt = (vectors * (1.0 / np.sqrt(values))[None, :]) @ vectors.conj().T
    return inverse_sqrt @ row_basis


def weighted_operator_relative_error(
    reference_basis: npt.NDArray[np.complex128],
    reference_weights: npt.NDArray[np.float64],
    observed_basis: npt.NDArray[np.complex128],
    observed_weights: npt.NDArray[np.float64],
) -> float:
    cross = reference_basis.conj() @ observed_basis.T
    inner = np.sum(
        reference_weights[:, None]
        * observed_weights[None, :]
        * np.square(np.abs(cross))
    )
    difference_squared = (
        np.sum(np.square(reference_weights))
        + np.sum(np.square(observed_weights))
        - 2.0 * inner
    )
    return float(
        np.sqrt(max(float(np.real(difference_squared)), 0.0))
        / np.linalg.norm(reference_weights)
    )


def fresh_model(manifest: dict[str, Any]) -> KLTParticleDetector3D:
    return KLTParticleDetector3D(
        None,
        manifest["particle_diameter_angstrom"],
        1.0 / manifest["voxel_size_angstrom"],
        manifest["truth_count"],
        legendre_order=150,
        max_order=manifest["max_order"],
        template_energy_fraction=manifest["template_energy_fraction"],
        max_templates=manifest["max_templates"],
        psd_patch_size=manifest["patch_size"],
        fredholm_radius=manifest["fredholm_radius_voxels"],
        template_side=manifest["template_side"],
        nms_radius=manifest["nms_radius_voxels"],
    )


def validate_templates(results_dir: Path, output_dir: Path) -> None:
    manifest = load_json(results_dir / "00_manifest.json")
    whitened_model = load_pickle(results_dir / "05_whitened_als.pkl")
    metadata = np.load(results_dir / "06_template_metadata.npz", allow_pickle=False)
    score_metadata = np.load(
        results_dir / "06b_block_qr_score_model.npz", allow_pickle=False
    )
    score_template_indices = (
        np.asarray(score_metadata["score_template_indices"], dtype=np.int64)
        if "score_template_indices" in score_metadata
        else np.arange(
            np.load(
                results_dir / "06b_block_qr_templates.npy",
                mmap_mode="r",
            ).shape[0],
            dtype=np.int64,
        )
    )
    raw_saved = np.load(results_dir / "06_templates.npy", mmap_mode="r")
    qr_saved = np.load(results_dir / "06b_block_qr_templates.npy", mmap_mode="r")

    model = fresh_model(manifest)
    radial_values, radial_functions, orders, particle_nodes = (
        model._solve_radial_modes(whitened_model.particle_psd)
    )
    template_m = np.asarray(metadata["template_m_values"], dtype=np.int64)
    compact_nonnegative_m = (
        np.all(template_m >= 0)
        and "template_multiplicities" in metadata
        and np.any(metadata["template_multiplicities"] == 2)
    )
    regenerated, _ = model.create_gpsf_templates(
        radial_values,
        radial_functions,
        orders,
        particle_nodes,
        nonnegative_m_only=compact_nonnegative_m,
    )

    radial_saved = np.asarray(metadata["radial_eigenvalues"])
    functions_saved = np.asarray(metadata["radial_eigenfunctions"])
    radial_errors = phase_aligned_relative_errors(model.eigfuncs, functions_saved)
    raw_errors = phase_aligned_relative_errors(regenerated, raw_saved)
    report: dict[str, Any] = {
        "radial_eigenvalue_max_abs_error": float(
            np.max(np.abs(model.radial_eigvals - radial_saved))
        ),
        "radial_eigenfunction_phase_aligned_max_relative_error": float(
            radial_errors.max()
        ),
        "raw_template_phase_aligned_max_relative_error": float(raw_errors.max()),
        "raw_template_phase_aligned_median_relative_error": float(
            np.median(raw_errors)
        ),
        "blocks": [],
    }
    del regenerated

    template_values = np.asarray(metadata["template_eigenvalues"], dtype=np.float64)
    template_orders = np.asarray(metadata["template_orders"], dtype=np.int64)
    saved_weights = np.asarray(score_metadata["score_weights"], dtype=np.float64)
    saved_signal = np.asarray(
        score_metadata["adjusted_template_eigenvalues"], dtype=np.float64
    )
    noise_variance = float(score_metadata["noise_variance"])
    score_template_indices = (
        np.asarray(score_metadata["score_template_indices"], dtype=np.int64)
        if "score_template_indices" in score_metadata
        else np.arange(raw_saved.shape[0], dtype=np.int64)
    )
    score_multiplicities = (
        np.asarray(score_metadata["score_multiplicities"], dtype=np.float64)
        if "score_multiplicities" in score_metadata
        else np.ones(qr_saved.shape[0], dtype=np.float64)
    )
    output_positions = {
        int(template_index): output_index
        for output_index, template_index in enumerate(score_template_indices)
    }
    reference_offset = 0.0

    for order, m_value, indices in block_indices(template_orders, template_m):
        if any(int(index) not in output_positions for index in indices):
            continue
        output_indices = np.asarray(
            [output_positions[int(index)] for index in indices],
            dtype=np.int64,
        )
        multiplicity = score_multiplicities[output_indices]
        if not np.all(multiplicity == multiplicity[0]):
            raise RuntimeError("one (ell,m) block has inconsistent multiplicities")
        reference_basis, weights, offset, signal = cpu_qr_score_block(
            np.asarray(raw_saved[indices]),
            template_values[indices],
            noise_variance,
        )
        observed_basis = np.asarray(
            qr_saved[output_indices], dtype=np.complex128
        ).reshape(indices.size, -1)
        observed_basis = orthonormalize_rows_from_gram(observed_basis)
        cross = reference_basis.conj() @ observed_basis.T
        singular_values = np.linalg.svd(cross, compute_uv=False)
        block_report = {
            "ell": int(order),
            "m": int(m_value),
            "width": int(indices.size),
            "minimum_principal_cosine": float(singular_values.min()),
            "maximum_principal_angle_degrees": float(
                np.degrees(np.arccos(np.clip(singular_values.min(), 0.0, 1.0)))
            ),
            "score_weight_max_abs_error": float(
                np.max(
                    np.abs(
                        multiplicity * weights
                        - saved_weights[output_indices]
                    )
                )
            ),
            "signal_eigenvalue_max_relative_error": float(
                np.max(
                    np.abs(signal - saved_signal[output_indices])
                    / np.maximum(np.abs(signal), np.finfo(np.float64).eps)
                )
            ),
            "weighted_operator_relative_frobenius_error": (
                weighted_operator_relative_error(
                    reference_basis,
                    multiplicity * weights,
                    observed_basis,
                    saved_weights[output_indices],
                )
            ),
        }
        report["blocks"].append(block_report)
        reference_offset += float(multiplicity[0]) * offset
        print(
            f"(ell,m)=({order:+d},{m_value:+d}) width={indices.size}: "
            f"min cos={singular_values.min():.9f}, "
            "weighted operator relerr="
            f"{block_report['weighted_operator_relative_frobenius_error']:.3e}"
        )

    report["score_offset_reference"] = reference_offset
    report["score_offset_saved"] = float(score_metadata["score_offset"])
    report["score_offset_abs_error"] = abs(
        reference_offset - float(score_metadata["score_offset"])
    )
    report["minimum_principal_cosine"] = min(
        block["minimum_principal_cosine"] for block in report["blocks"]
    )
    report["maximum_weighted_operator_relative_frobenius_error"] = max(
        block["weighted_operator_relative_frobenius_error"]
        for block in report["blocks"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(report, output_dir / "template_validation.json")
    print(f"Saved {output_dir / 'template_validation.json'}")


def load_scoring_inputs(
    results_dir: Path,
    output_dir: Path,
    region_start: tuple[int, int, int],
    devices: tuple[jax.Device, ...],
) -> tuple[
    dict[str, Any],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.generic],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    float,
]:
    manifest = load_json(results_dir / "00_manifest.json")
    core_shape = tuple(int(value) for value in manifest["core_shape"])
    whitening_radius = int(
        (np.load(results_dir / "03_whitening_filter.npy").shape[0] - 1) // 2
    )
    template_radius = int(
        (np.load(results_dir / "06b_block_qr_templates.npy", mmap_mode="r").shape[1]
         - 1)
        // 2
    )
    total_halo = whitening_radius + template_radius + 1
    crop_path = output_dir / "loaded_subvolume.npy"
    output_dir.mkdir(parents=True, exist_ok=True)
    if crop_path.is_file():
        loaded = np.load(crop_path, allow_pickle=False)
    else:
        with MrcVolumeSource(manifest["input"]) as source:
            processor = MultiGPUSubvolumeProcessor(
                source,
                domain_shape=source.shape,
                core_shape=core_shape,
                devices=devices,
                boundary_mode="constant",
            )
            region = SpatialRegion(
                start=region_start,
                stop=tuple(
                    start + size
                    for start, size in zip(region_start, core_shape, strict=True)
                ),
            )
            loaded = processor.load_region(region, total_halo)
        np.save(crop_path, loaded, allow_pickle=False)
    expected_shape = tuple(size + 2 * total_halo for size in core_shape)
    if loaded.shape != expected_shape:
        raise ValueError(f"loaded crop has {loaded.shape}, expected {expected_shape}")

    metadata = np.load(results_dir / "06_template_metadata.npz", allow_pickle=False)
    score_metadata = np.load(
        results_dir / "06b_block_qr_score_model.npz", allow_pickle=False
    )
    manifest = {
        **manifest,
        "whitening_radius": whitening_radius,
        "template_radius": template_radius,
        "region_start": list(region_start),
    }
    return (
        manifest,
        np.asarray(loaded, dtype=np.float32),
        np.asarray(np.load(results_dir / "03_whitening_filter.npy"), dtype=np.float32),
        np.load(results_dir / "06b_block_qr_templates.npy", mmap_mode="r"),
        np.asarray(score_metadata["score_weights"], dtype=np.float64),
        np.asarray(metadata["template_orders"], dtype=np.int64)[
            score_template_indices
        ],
        np.asarray(metadata["template_m_values"], dtype=np.int64)[
            score_template_indices
        ],
        float(score_metadata["score_offset"]),
    )


def score_single(
    results_dir: Path,
    output_dir: Path,
    region_start: tuple[int, int, int],
) -> None:
    devices = tuple(jax.devices())
    if len(devices) != 1:
        raise RuntimeError(
            "score-single requires exactly one visible device; set CUDA_VISIBLE_DEVICES"
        )
    (
        manifest,
        loaded,
        whitening,
        templates,
        weights,
        orders,
        m_values,
        offset,
    ) = load_scoring_inputs(results_dir, output_dir, region_start, devices)
    blocks = block_indices(orders, m_values)
    output_shape = tuple(int(size) + 2 for size in manifest["core_shape"])
    block_scores = np.empty((len(blocks), *output_shape), dtype=np.float32)
    compiled_by_width: dict[int, Any] = {}
    for block_number, (order, m_value, indices) in enumerate(blocks):
        width = int(indices.size)
        if width not in compiled_by_width:
            configured = partial(
                compute_klt_score_block,
                core_shape=tuple(manifest["core_shape"]),
                whitening_radius=manifest["whitening_radius"],
                score_halo=1,
            )
            compiled_by_width[width] = jax.jit(configured)
        result = compiled_by_width[width](
            loaded,
            whitening,
            np.asarray(templates[indices], dtype=np.complex64),
            np.asarray(weights[indices], dtype=np.float32),
            np.float32(0.0),
        )
        block_scores[block_number] = np.asarray(result)
        print(f"single block {block_number + 1}/{len(blocks)}: ({order},{m_value})")
    total = np.sum(block_scores, axis=0, dtype=np.float64).astype(np.float32) - offset
    np.save(output_dir / "single_block_scores.npy", block_scores, allow_pickle=False)
    np.save(output_dir / "single_total_score.npy", total, allow_pickle=False)
    print("Saved single-device dense scores")


def shard_templates(
    templates: npt.NDArray[np.generic],
    weights: npt.NDArray[np.float64],
    devices: tuple[jax.Device, ...],
    batch_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    device_count = len(devices)
    per_device = (templates.shape[0] + device_count - 1) // device_count
    padded = ((per_device + batch_size - 1) // batch_size) * batch_size
    template_shards = []
    normalization_shards = []
    weight_shards = []
    for device_index in range(device_count):
        start = device_index * per_device
        stop = min(start + per_device, templates.shape[0])
        count = max(stop - start, 0)
        template_shard = np.zeros(
            (padded, *templates.shape[1:]), dtype=np.complex64
        )
        normalization = np.zeros(padded, dtype=np.float32)
        weight_shard = np.zeros(padded, dtype=np.float32)
        if count:
            template_shard[:count] = np.asarray(
                templates[start:stop], dtype=np.complex64
            )
            normalization[:count] = 1.0
            weight_shard[:count] = weights[start:stop]
        template_shards.append(template_shard)
        normalization_shards.append(normalization)
        weight_shards.append(weight_shard)
    return (
        jax.device_put_sharded(template_shards, devices),
        jax.device_put_sharded(normalization_shards, devices),
        jax.device_put_sharded(weight_shards, devices),
    )


def sharded_dense_score(
    loaded: npt.NDArray[np.float32],
    whitening: npt.NDArray[np.float32],
    templates: npt.NDArray[np.generic],
    weights: npt.NDArray[np.float64],
    offset: float,
    manifest: dict[str, Any],
    devices: tuple[jax.Device, ...],
    batch_size: int,
) -> npt.NDArray[np.float32]:
    device_templates, device_normalization, device_weights = shard_templates(
        templates, weights, devices, batch_size
    )
    configured = partial(
        compute_fused_klt_score_shard,
        core_shape=tuple(manifest["core_shape"]),
        whitening_radius=manifest["whitening_radius"],
        template_radius=manifest["template_radius"],
        score_halo=1,
        template_batch_size=batch_size,
        axis_name=_TEMPLATE_AXIS_NAME,
    )
    mapped = jax.pmap(
        configured,
        axis_name=_TEMPLATE_AXIS_NAME,
        in_axes=(None, None, 0, 0, 0, None),
        devices=devices,
    )
    result = mapped(
        loaded,
        whitening,
        device_templates,
        device_normalization,
        device_weights,
        np.float32(offset),
    )
    return np.asarray(result[0], dtype=np.float32)


def score_sharded(
    results_dir: Path,
    output_dir: Path,
    region_start: tuple[int, int, int],
    batch_size: int,
) -> None:
    devices = tuple(jax.devices())
    if len(devices) < 2:
        raise RuntimeError("score-sharded requires at least two visible devices")
    (
        manifest,
        loaded,
        whitening,
        templates,
        weights,
        orders,
        m_values,
        offset,
    ) = load_scoring_inputs(results_dir, output_dir, region_start, devices)
    production_total = sharded_dense_score(
        loaded,
        whitening,
        templates,
        weights,
        offset,
        manifest,
        devices,
        batch_size,
    )
    np.save(
        output_dir / "sharded_production_total_score.npy",
        production_total,
        allow_pickle=False,
    )
    print("saved production contiguous-shard total")

    blocks = block_indices(orders, m_values)
    output_shape = tuple(int(size) + 2 for size in manifest["core_shape"])
    block_scores = np.empty((len(blocks), *output_shape), dtype=np.float32)
    for block_number, (order, m_value, indices) in enumerate(blocks):
        block_scores[block_number] = sharded_dense_score(
            loaded,
            whitening,
            templates[indices],
            weights[indices],
            0.0,
            manifest,
            devices,
            batch_size,
        )
        print(f"sharded block {block_number + 1}/{len(blocks)}: ({order},{m_value})")
    block_total = (
        np.sum(block_scores, axis=0, dtype=np.float64).astype(np.float32) - offset
    )
    np.save(output_dir / "sharded_block_scores.npy", block_scores, allow_pickle=False)
    np.save(output_dir / "sharded_block_total_score.npy", block_total, allow_pickle=False)
    print("Saved sharded dense scores")


def array_comparison(
    reference: npt.NDArray[np.generic],
    observed: npt.NDArray[np.generic],
) -> dict[str, float | list[int]]:
    reference64 = np.asarray(reference, dtype=np.float64)
    observed64 = np.asarray(observed, dtype=np.float64)
    difference = observed64 - reference64
    reference_norm = np.linalg.norm(reference64.reshape(-1))
    centered_reference = reference64.reshape(-1) - np.mean(reference64)
    centered_observed = observed64.reshape(-1) - np.mean(observed64)
    return {
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "relative_l2_error": float(
            np.linalg.norm(difference.reshape(-1)) / max(reference_norm, 1e-30)
        ),
        "correlation": float(
            np.vdot(centered_reference, centered_observed).real
            / max(
                np.linalg.norm(centered_reference)
                * np.linalg.norm(centered_observed),
                1e-30,
            )
        ),
        "reference_argmax": list(
            map(int, np.unravel_index(np.argmax(reference64), reference64.shape))
        ),
        "observed_argmax": list(
            map(int, np.unravel_index(np.argmax(observed64), observed64.shape))
        ),
    }


def compare_candidates(
    score: npt.NDArray[np.generic],
    manifest: dict[str, Any],
    region_start: tuple[int, int, int],
    *,
    candidate_capacity: int = 4096,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float32], int]:
    configured = jax.jit(
        partial(
            extract_score_candidates,
            core_shape=tuple(manifest["core_shape"]),
            source_shape=tuple(manifest["volume_shape_zyx"]),
            template_radius=int(manifest["template_side"]) // 2,
            candidate_capacity=candidate_capacity,
        )
    )
    coordinates, scores, count = configured(
        np.asarray(score),
        np.asarray(region_start, dtype=np.int32),
    )
    return np.asarray(coordinates), np.asarray(scores), int(np.asarray(count))


def compare_scores(
    results_dir: Path,
    output_dir: Path,
    region_start: tuple[int, int, int],
) -> None:
    single_blocks = np.load(output_dir / "single_block_scores.npy", mmap_mode="r")
    sharded_blocks = np.load(output_dir / "sharded_block_scores.npy", mmap_mode="r")
    report: dict[str, Any] = {
        "blocks": [
            {"block_index": index, **array_comparison(single_blocks[index], sharded_blocks[index])}
            for index in range(single_blocks.shape[0])
        ]
    }
    single_total = np.load(output_dir / "single_total_score.npy", mmap_mode="r")
    block_total = np.load(
        output_dir / "sharded_block_total_score.npy", mmap_mode="r"
    )
    production_total = np.load(
        output_dir / "sharded_production_total_score.npy", mmap_mode="r"
    )
    report["single_vs_sharded_block_total"] = array_comparison(
        single_total, block_total
    )
    report["single_vs_sharded_production_total"] = array_comparison(
        single_total, production_total
    )
    report["sharded_block_vs_production_total"] = array_comparison(
        block_total, production_total
    )
    manifest = load_json(results_dir / "00_manifest.json")
    single_coordinates, single_candidate_scores, single_count = compare_candidates(
        single_total, manifest, region_start
    )
    sharded_coordinates, sharded_candidate_scores, sharded_count = compare_candidates(
        production_total, manifest, region_start
    )
    coordinate_matches = np.all(
        single_coordinates == sharded_coordinates,
        axis=1,
    )
    single_coordinate_set = {tuple(row) for row in single_coordinates.tolist()}
    sharded_coordinate_set = {tuple(row) for row in sharded_coordinates.tolist()}
    finite = np.isfinite(single_candidate_scores) & np.isfinite(
        sharded_candidate_scores
    )
    report["candidate_comparison"] = {
        "single_local_maximum_count": single_count,
        "sharded_local_maximum_count": sharded_count,
        "top4096_coordinate_rankwise_match_count": int(
            np.sum(coordinate_matches)
        ),
        "top4096_coordinate_rankwise_match_fraction": float(
            np.mean(coordinate_matches)
        ),
        "top4096_coordinate_set_intersection": len(
            single_coordinate_set & sharded_coordinate_set
        ),
        "top4096_coordinate_sets_equal": (
            single_coordinate_set == sharded_coordinate_set
        ),
        "finite_score_maximum_absolute_error": float(
            np.max(
                np.abs(
                    single_candidate_scores[finite]
                    - sharded_candidate_scores[finite]
                )
            )
        ),
    }
    save_json(report, output_dir / "score_validation.json")
    print(json.dumps(report["single_vs_sharded_production_total"], indent=2))
    print(f"Saved {output_dir / 'score_validation.json'}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    region_start = tuple(int(value) for value in args.region_start)
    if args.mode == "templates":
        validate_templates(args.results_dir, args.output_dir)
    elif args.mode == "score-single":
        score_single(args.results_dir, args.output_dir, region_start)
    elif args.mode == "score-sharded":
        score_sharded(
            args.results_dir,
            args.output_dir,
            region_start,
            args.template_batch_size,
        )
    else:
        compare_scores(args.results_dir, args.output_dir, region_start)


if __name__ == "__main__":
    main()
