"""Measure angular-order convergence of the EMPIAR-10045 KLT spectrum.

This experiment uses the saved whitened particle RPSD and solves the radial
Fredholm problem for a deliberately high spherical-harmonic ceiling. It does
not construct Cartesian templates or rescore the tomogram. The outputs answer
three separate questions:

1. How much of the full covariance trace is exposed by each angular ceiling?
2. How many globally ranked ``(ell, n)`` modes reach a target energy fraction?
3. How much energy remains when complete ``m=-ell,...,ell`` multiplets are
   constrained by a finite Cartesian-template budget?
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import numpy.typing as npt
from scipy.special import roots_legendre
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from kltpicker_3d.fredholm_solver import (
    INVERSE_FOURIER_NORMALIZATION_3D,
    solve_radial_fredholm_equation,
)
from kltpicker_3d.utils import trigonometric_interpolation


DEFAULT_RESULTS = REPOSITORY_ROOT / "results/empiar-10045-bandpass-block-qr"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "results/empiar-10045-max-order-analysis"
DEFAULT_THRESHOLDS = (0.80, 0.90, 0.95, 0.99)


def parse_args() -> argparse.Namespace:
    """Parse spectrum-analysis settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-order",
        type=int,
        default=180,
        help="Exclusive ell ceiling; default solves ell=0,...,179",
    )
    parser.add_argument("--legendre-order", type=int, default=150)
    parser.add_argument("--energy-fraction", type=float, default=0.99)
    parser.add_argument("--template-cap", type=int, default=1000)
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Ignore a compatible cached angular spectrum",
    )
    return parser.parse_args()


def load_pickle(path: Path) -> Any:
    """Load one trusted local experiment checkpoint."""
    with path.open("rb") as stream:
        return pickle.load(stream)


def save_json(value: dict[str, Any], path: Path) -> None:
    """Write deterministic, human-readable JSON."""
    with path.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def array_digest(values: npt.ArrayLike) -> str:
    """Return a stable digest for cache validation."""
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def solve_angular_spectrum(
    particle_psd_nodes: npt.NDArray[np.float64],
    *,
    max_order: int,
    support_radius: float,
    bandlimit: float,
    legendre_order: int,
) -> npt.NDArray[np.float64]:
    """Return one padded radial eigenvalue row for every angular order."""
    eigenvalues = np.zeros((max_order, legendre_order), dtype=np.float64)
    for ell in tqdm(range(max_order), desc="Fredholm angular sweep", unit="ell"):
        values, _, _ = solve_radial_fredholm_equation(
            particle_psd_nodes,
            N=ell,
            a=support_radius,
            c=bandlimit,
            K=legendre_order,
        )
        eigenvalues[ell] = np.maximum(np.real(values), 0.0)
    return eigenvalues


def sorted_radial_modes(
    eigenvalues: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
]:
    """Return positive radial eigenmodes globally sorted by eigenvalue."""
    ell_grid, radial_grid = np.indices(eigenvalues.shape)
    positive = eigenvalues > 0
    values = eigenvalues[positive]
    orders = ell_grid[positive].astype(np.int64)
    radial_indices = radial_grid[positive].astype(np.int64)
    descending = np.argsort(values, kind="stable")[::-1]
    return values[descending], orders[descending], radial_indices[descending]


def retained_prefix(
    values: npt.NDArray[np.float64],
    orders: npt.NDArray[np.int64],
    *,
    available_energy: float,
    energy_fraction: float,
    template_cap: int | None,
) -> int:
    """Return the complete-multiplet prefix allowed by energy and memory."""
    multiplicities = 2 * orders + 1
    cumulative_energy = np.cumsum(values * multiplicities)
    energy_count = int(
        np.searchsorted(
            cumulative_energy,
            energy_fraction * available_energy,
            side="left",
        )
        + 1
    )
    if template_cap is None:
        return energy_count
    cumulative_templates = np.cumsum(multiplicities)
    cap_count = int(np.searchsorted(cumulative_templates, template_cap, side="right"))
    return min(energy_count, cap_count)


def write_angular_csv(
    path: Path,
    eigenvalues: npt.NDArray[np.float64],
    expected_trace: float,
) -> None:
    """Write per-order and cumulative covariance-energy diagnostics."""
    multiplicities = 2 * np.arange(eigenvalues.shape[0]) + 1
    block_energy = multiplicities * np.sum(eigenvalues, axis=1)
    cumulative = np.cumsum(block_energy)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "ell",
                "m_multiplicity",
                "positive_radial_modes",
                "block_energy",
                "block_trace_fraction",
                "cumulative_energy",
                "cumulative_trace_fraction",
            )
        )
        for ell in range(eigenvalues.shape[0]):
            writer.writerow(
                (
                    ell,
                    multiplicities[ell],
                    int(np.count_nonzero(eigenvalues[ell] > 0)),
                    block_energy[ell],
                    block_energy[ell] / expected_trace,
                    cumulative[ell],
                    cumulative[ell] / expected_trace,
                )
            )


def write_global_modes_csv(
    path: Path,
    values: npt.NDArray[np.float64],
    orders: npt.NDArray[np.int64],
    radial_indices: npt.NDArray[np.int64],
    expected_trace: float,
) -> None:
    """Write the globally ranked radial modes and expanded-template cost."""
    multiplicities = 2 * orders + 1
    cumulative_templates = np.cumsum(multiplicities)
    cumulative_energy = np.cumsum(values * multiplicities)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "rank",
                "ell",
                "radial_index_n",
                "eigenvalue",
                "m_multiplicity",
                "cumulative_templates",
                "cumulative_trace_fraction",
            )
        )
        for index in range(values.size):
            writer.writerow(
                (
                    index + 1,
                    orders[index],
                    radial_indices[index],
                    values[index],
                    multiplicities[index],
                    cumulative_templates[index],
                    cumulative_energy[index] / expected_trace,
                )
            )


def ceiling_analysis(
    eigenvalues: npt.NDArray[np.float64],
    expected_trace: float,
    energy_fraction: float,
    template_cap: int,
) -> list[dict[str, int | float]]:
    """Evaluate spectral exposure and retained energy for every ell ceiling."""
    rows = []
    for order_count in range(1, eigenvalues.shape[0] + 1):
        values, orders, _ = sorted_radial_modes(eigenvalues[:order_count])
        multiplicities = 2 * orders + 1
        available_energy = float(np.sum(values * multiplicities))
        retained_count = retained_prefix(
            values,
            orders,
            available_energy=available_energy,
            energy_fraction=energy_fraction,
            template_cap=template_cap,
        )
        retained_energy = float(
            np.sum(values[:retained_count] * multiplicities[:retained_count])
        )
        rows.append(
            {
                "max_order": order_count,
                "largest_ell": order_count - 1,
                "available_trace_fraction": available_energy / expected_trace,
                "retained_trace_fraction": retained_energy / expected_trace,
                "retained_fraction_of_available": retained_energy / available_energy,
                "retained_radial_modes": retained_count,
                "retained_templates": int(np.sum(multiplicities[:retained_count])),
            }
        )
    return rows


def write_ceiling_csv(path: Path, rows: list[dict[str, int | float]]) -> None:
    """Write max-order convergence after energy and template truncation."""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def threshold_crossings(
    fractions: npt.NDArray[np.float64],
) -> dict[str, dict[str, int | float] | None]:
    """Return the first angular ceiling that reaches standard trace targets."""
    crossings: dict[str, dict[str, int | float] | None] = {}
    for target in DEFAULT_THRESHOLDS:
        indices = np.flatnonzero(fractions >= target)
        crossings[f"{target:.2f}"] = (
            None
            if indices.size == 0
            else {
                "max_order": int(indices[0] + 1),
                "largest_ell": int(indices[0]),
                "trace_fraction": float(fractions[indices[0]]),
            }
        )
    return crossings


def global_mode_crossings(
    values: npt.NDArray[np.float64],
    orders: npt.NDArray[np.int64],
    expected_trace: float,
) -> dict[str, dict[str, int | float] | None]:
    """Return radial-mode and Cartesian-template costs for trace targets."""
    multiplicities = 2 * orders + 1
    cumulative_energy = np.cumsum(values * multiplicities)
    cumulative_templates = np.cumsum(multiplicities)
    crossings: dict[str, dict[str, int | float] | None] = {}
    for target in DEFAULT_THRESHOLDS:
        indices = np.flatnonzero(cumulative_energy / expected_trace >= target)
        crossings[f"{target:.2f}"] = (
            None
            if indices.size == 0
            else {
                "radial_modes": int(indices[0] + 1),
                "expanded_templates": int(cumulative_templates[indices[0]]),
                "largest_selected_ell": int(np.max(orders[: indices[0] + 1])),
                "trace_fraction": float(
                    cumulative_energy[indices[0]] / expected_trace
                ),
            }
        )
    return crossings


def plot_results(
    output_dir: Path,
    eigenvalues: npt.NDArray[np.float64],
    expected_trace: float,
    ceiling_rows: list[dict[str, int | float]],
    values: npt.NDArray[np.float64],
    orders: npt.NDArray[np.int64],
    template_cap: int,
) -> None:
    """Create a compact three-panel diagnostic of all truncation effects."""
    angular_orders = np.arange(eigenvalues.shape[0])
    multiplicities = 2 * angular_orders + 1
    block_fraction = multiplicities * np.sum(eigenvalues, axis=1) / expected_trace
    cumulative_fraction = np.cumsum(block_fraction)

    figure, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    axes[0].plot(angular_orders, cumulative_fraction, linewidth=2, label="cumulative")
    axes[0].plot(angular_orders, block_fraction, linewidth=1.2, label="per ell")
    for target in DEFAULT_THRESHOLDS:
        axes[0].axhline(target, color="0.75", linewidth=0.8, linestyle="--")
    axes[0].set(
        xlabel=r"largest included $\ell$",
        ylabel="fraction of covariance trace",
        title="Angular-spectrum convergence",
        ylim=(0, 1.03),
    )
    axes[0].legend()

    max_orders = np.asarray([row["max_order"] for row in ceiling_rows])
    available = np.asarray([row["available_trace_fraction"] for row in ceiling_rows])
    retained = np.asarray([row["retained_trace_fraction"] for row in ceiling_rows])
    axes[1].plot(max_orders, available, linewidth=2, label="spectrum exposed")
    axes[1].plot(
        max_orders,
        retained,
        linewidth=2,
        label=f"retained with cap={template_cap}",
    )
    axes[1].set(
        xlabel="max_order (exclusive)",
        ylabel="fraction of covariance trace",
        title="Ceiling versus retained model",
        ylim=(0, 1.03),
    )
    axes[1].legend()

    mode_multiplicities = 2 * orders + 1
    cumulative_templates = np.cumsum(mode_multiplicities)
    cumulative_mode_energy = np.cumsum(values * mode_multiplicities) / expected_trace
    axes[2].plot(cumulative_templates, cumulative_mode_energy, linewidth=2)
    axes[2].axvline(template_cap, color="#CC3311", linestyle="--", label="template cap")
    for target in DEFAULT_THRESHOLDS:
        axes[2].axhline(target, color="0.75", linewidth=0.8, linestyle="--")
    axes[2].set(
        xlabel="expanded Cartesian templates",
        ylabel="fraction of covariance trace",
        title="Global eigenvalue truncation",
        xscale="log",
        ylim=(0, 1.03),
    )
    axes[2].legend()

    figure.suptitle("EMPIAR-10045 tomogram 08: spherical KLT order convergence")
    figure.tight_layout()
    figure.savefig(output_dir / "max_order_convergence.png", dpi=220)
    figure.savefig(output_dir / "max_order_convergence.pdf")
    plt.close(figure)


def main() -> None:
    """Run or resume the high-order EMPIAR covariance-spectrum analysis."""
    args = parse_args()
    if args.max_order < 1 or args.legendre_order < 1:
        raise ValueError("max-order and legendre-order must be positive")
    if not 0 < args.energy_fraction <= 1:
        raise ValueError("energy-fraction must lie in (0, 1]")
    if args.template_cap < 1:
        raise ValueError("template-cap must be positive")

    manifest = json.loads((args.results_dir / "00_manifest.json").read_text())
    extraction = load_pickle(args.results_dir / "04_whitened_patch_rpsds.pkl")
    whitened_model = load_pickle(args.results_dir / "05_whitened_als.pkl")
    radial_points = np.asarray(extraction.radial_points, dtype=np.float64)
    particle_psd = np.asarray(whitened_model.particle_psd, dtype=np.float64)
    support_radius = float(manifest["fredholm_radius_voxels"])
    bandlimit = float(radial_points[-1])

    nodes, weights = roots_legendre(args.legendre_order)
    frequency_nodes = (bandlimit / 2) * (nodes + 1)
    particle_psd_nodes = np.maximum(
        np.asarray(
            trigonometric_interpolation(
                radial_points,
                particle_psd,
                frequency_nodes,
            ),
            dtype=np.float64,
        ),
        0.0,
    )
    covariance_at_zero = (
        4
        * np.pi
        * INVERSE_FOURIER_NORMALIZATION_3D
        * (bandlimit / 2)
        * np.sum(weights * particle_psd_nodes * frequency_nodes**2)
    )
    expected_trace = 4 * np.pi * support_radius**3 / 3 * covariance_at_zero

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "angular_eigenvalues.npz"
    digest = array_digest(particle_psd_nodes)
    eigenvalues: npt.NDArray[np.float64]
    if cache_path.is_file() and not args.recompute:
        cache = np.load(cache_path, allow_pickle=False)
        compatible = (
            int(cache["max_order"]) == args.max_order
            and int(cache["legendre_order"]) == args.legendre_order
            and float(cache["support_radius"]) == support_radius
            and float(cache["bandlimit"]) == bandlimit
            and str(cache["particle_psd_digest"]) == digest
        )
        if compatible:
            eigenvalues = np.asarray(cache["eigenvalues"], dtype=np.float64)
            print(f"Using cached angular spectrum: {cache_path}")
        else:
            raise ValueError(
                f"incompatible cache at {cache_path}; rerun with --recompute"
            )
    else:
        eigenvalues = solve_angular_spectrum(
            particle_psd_nodes,
            max_order=args.max_order,
            support_radius=support_radius,
            bandlimit=bandlimit,
            legendre_order=args.legendre_order,
        )
        np.savez(
            cache_path,
            eigenvalues=eigenvalues,
            max_order=args.max_order,
            legendre_order=args.legendre_order,
            support_radius=support_radius,
            bandlimit=bandlimit,
            particle_psd_digest=digest,
        )

    angular_multiplicities = 2 * np.arange(args.max_order) + 1
    block_energy = angular_multiplicities * np.sum(eigenvalues, axis=1)
    angular_fraction = np.cumsum(block_energy) / expected_trace
    values, orders, radial_indices = sorted_radial_modes(eigenvalues)
    mode_multiplicities = 2 * orders + 1
    available_energy = float(np.sum(values * mode_multiplicities))
    retained_count = retained_prefix(
        values,
        orders,
        available_energy=available_energy,
        energy_fraction=args.energy_fraction,
        template_cap=args.template_cap,
    )
    retained_energy = float(
        np.sum(values[:retained_count] * mode_multiplicities[:retained_count])
    )
    retained_templates = int(np.sum(mode_multiplicities[:retained_count]))

    ceiling_rows = ceiling_analysis(
        eigenvalues,
        expected_trace,
        args.energy_fraction,
        args.template_cap,
    )
    write_angular_csv(
        args.output_dir / "angular_order_energy.csv",
        eigenvalues,
        expected_trace,
    )
    write_global_modes_csv(
        args.output_dir / "global_radial_modes.csv",
        values,
        orders,
        radial_indices,
        expected_trace,
    )
    write_ceiling_csv(args.output_dir / "max_order_tradeoff.csv", ceiling_rows)
    plot_results(
        args.output_dir,
        eigenvalues,
        expected_trace,
        ceiling_rows,
        values,
        orders,
        args.template_cap,
    )

    summary = {
        "source_results": str(args.results_dir.resolve()),
        "particle_psd_digest": digest,
        "legendre_order": args.legendre_order,
        "max_order_exclusive": args.max_order,
        "largest_solved_ell": args.max_order - 1,
        "support_radius_voxels": support_radius,
        "bandlimit_radians_per_voxel": bandlimit,
        "space_bandwidth_product": support_radius * bandlimit,
        "inverse_fourier_normalization_3d": INVERSE_FOURIER_NORMALIZATION_3D,
        "particle_variance_from_solver_quadrature": covariance_at_zero,
        "expected_covariance_trace": expected_trace,
        "solved_covariance_trace": available_energy,
        "solved_trace_fraction": available_energy / expected_trace,
        "angular_ceiling_crossings": threshold_crossings(angular_fraction),
        "global_mode_crossings": global_mode_crossings(
            values,
            orders,
            expected_trace,
        ),
        "requested_energy_fraction": args.energy_fraction,
        "template_cap": args.template_cap,
        "retained_radial_modes": retained_count,
        "retained_templates": retained_templates,
        "retained_fraction_of_solved_spectrum": retained_energy / available_energy,
        "retained_fraction_of_expected_trace": retained_energy / expected_trace,
        "cap_is_binding": bool(
            retained_energy / available_energy
            < args.energy_fraction - np.finfo(np.float64).eps
        ),
    }
    save_json(summary, args.output_dir / "summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved detailed analysis to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
