"""Prepare KLT picks for the legacy RELION subtomogram workflow.

EMPIAR-10045 predates the modern RELION tomography data model.  The archive
contains aligned tilt stacks, final tilt angles, acquisition-order/dose files,
trial images, and CTFFIND results.  This script converts ranked KLT picks into
the legacy RELION layout and recreates the per-particle 3D-CTF metadata without
rerunning CTFFIND or requiring IMOD.

The generated extraction and CTF-reconstruction scripts are deliberately not
executed.  A full top-10N run writes tens of GiB even after rescaling.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex

import mrcfile
import numpy as np


DEFAULT_DATASET_ROOT = Path(
    "/data/yoelsh/datasets/10045/pristine/data/ribosomes"
)
DEFAULT_CANDIDATES = Path(
    "results/empiar-10045-recall-nms-nonstationarity/"
    "ranked_nms_0.5D_top10N.npy"
)
DEFAULT_OUTPUT = Path("results/empiar-10045-relion-legacy")
TOMOGRAM_NAME = "IS002_291013_008"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--box-size", type=int, default=200)
    parser.add_argument(
        "--scaled-size",
        type=int,
        default=128,
        help="Fourier-rescaled particle box used for classification",
    )
    parser.add_argument("--particle-diameter", type=float, default=350.0)
    parser.add_argument(
        "--legacy-pixel-size",
        type=float,
        default=10000.0 * 11.57 / 53000.0,
        help="Pixel size used by the deposited legacy RELION workflow",
    )
    parser.add_argument("--low-tilt-limit", type=float, default=30.0)
    parser.add_argument("--dose-bfactor", type=float, default=4.0)
    parser.add_argument(
        "--max-picks",
        type=int,
        default=None,
        help="Optional ranked-prefix limit, useful for smoke tests",
    )
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def replace_symlink(link: Path, target: Path) -> None:
    """Create a relative symlink, replacing only a pre-existing symlink."""
    target = target.resolve()
    if link.is_symlink():
        if link.resolve() == target:
            return
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"refusing to replace non-symlink: {link}")
    link.symlink_to(os.path.relpath(target, start=link.parent))


def read_star_table(path: Path) -> tuple[list[str], list[list[str]]]:
    labels: list[str] = []
    rows: list[list[str]] = []
    in_loop = False
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line == "loop_":
            in_loop = True
            continue
        if in_loop and line.startswith("_"):
            labels.append(line.split()[0])
            continue
        if labels and not line.startswith(("data_", "loop_", "_")):
            fields = shlex.split(line)
            if len(fields) != len(labels):
                raise ValueError(
                    f"{path}: expected {len(labels)} fields, found {len(fields)}"
                )
            rows.append(fields)
    if not labels or not rows:
        raise ValueError(f"no STAR loop found in {path}")
    return labels, rows


def deposited_defocus_per_tilt(
    ctf_star: Path,
    tilts: np.ndarray,
    low_tilt_limit: float,
) -> np.ndarray:
    labels, rows = read_star_table(ctf_star)
    try:
        name_column = labels.index("_rlnMicrographName")
        defocus_column = labels.index("_rlnDefocusU")
    except ValueError as error:
        raise ValueError(f"missing required label in {ctf_star}: {error}") from error

    rows = [row for row in rows if TOMOGRAM_NAME in row[name_column]]
    defocus_u = np.asarray(
        [float(row[defocus_column]) for row in rows], dtype=np.float64
    )
    if defocus_u.size != 2 * tilts.size:
        raise ValueError(
            f"expected two trial-image CTF estimates per tilt: "
            f"{defocus_u.size} estimates for {tilts.size} tilts"
        )

    # This is the rule used by relion_prepare_subtomo: average the two trial
    # estimates, then replace unreliable high-tilt estimates by the mean of
    # estimates with |tilt| below the configured limit.
    defocus = defocus_u.reshape(-1, 2).mean(axis=1)
    low_tilt = np.abs(tilts) < low_tilt_limit
    if not np.any(low_tilt):
        raise ValueError("low-tilt limit selects no CTF estimates")
    defocus[np.abs(tilts) > low_tilt_limit] = defocus[low_tilt].mean()
    return defocus


def write_coordinate_star(path: Path, candidates: np.ndarray) -> None:
    with path.open("w") as output:
        output.write(
            "data_coordinates\n\n"
            "loop_\n"
            "_rlnCoordinateX #1\n"
            "_rlnCoordinateY #2\n"
            "_rlnCoordinateZ #3\n"
            "_rlnAutopickFigureOfMerit #4\n"
        )
        for z, y, x, score in candidates:
            output.write(f"{x:.6f} {y:.6f} {z:.6f} {score:.9g}\n")


def write_coordinate_text(path: Path, candidates: np.ndarray) -> None:
    xyz = candidates[:, [2, 1, 0]]
    np.savetxt(path, xyz, fmt="%.6f", delimiter="\t")


def particle_defocus(
    xyz: np.ndarray,
    tilts: np.ndarray,
    base_defocus: np.ndarray,
    tomogram_xyz: tuple[int, int, int],
    pixel_size: float,
) -> np.ndarray:
    xdim, _, zdim = tomogram_xyz
    x = xyz[:, 0, None]
    z = xyz[:, 2, None]
    radians = np.deg2rad(tilts)[None, :]
    centered_x = (x - xdim // 2) * pixel_size
    centered_z = (z - zdim // 2) * pixel_size
    projected_x = centered_x * np.cos(radians) + centered_z * np.sin(radians)
    return base_defocus[None, :] + projected_x * np.sin(radians)


def matched_doses(tilts: np.ndarray, order: np.ndarray) -> np.ndarray:
    order_angles = order[:, 0]
    indices = np.argmin(np.abs(tilts[:, None] - order_angles[None, :]), axis=1)
    differences = np.abs(tilts - order_angles[indices])
    # The order file contains nominal integer angles, whereas .tlt contains
    # refined IMOD angles. RELION's legacy preparer permits roughly one full
    # tilt step here. Require a one-to-one nearest-angle mapping as the more
    # important guard against assigning the wrong accumulated dose.
    if np.unique(indices).size != tilts.size:
        raise ValueError("tilt-to-order matching is not one-to-one")
    median_step = float(np.median(np.diff(np.sort(order_angles))))
    tolerance = abs(median_step) / 2.0 + 0.25
    if np.max(differences) > tolerance:
        raise ValueError(
            f"could not match tilt and order files; maximum difference "
            f"is {np.max(differences):.3f} degrees (limit {tolerance:.3f})"
        )
    return order[indices, 1]


def write_particle_ctf_star(
    path: Path,
    tilts: np.ndarray,
    defocus: np.ndarray,
    bfactor: np.ndarray,
) -> None:
    scale = np.cos(np.abs(np.deg2rad(tilts)))
    with path.open("w") as output:
        output.write(
            "data_images\n\n"
            "loop_\n"
            "_rlnDefocusU #1\n"
            "_rlnVoltage #2\n"
            "_rlnSphericalAberration #3\n"
            "_rlnAmplitudeContrast #4\n"
            "_rlnAngleRot #5\n"
            "_rlnAngleTilt #6\n"
            "_rlnAnglePsi #7\n"
            "_rlnCtfBfactor #8\n"
            "_rlnCtfScalefactor #9\n"
        )
        for angle, particle_defocus_u, dose_b, tilt_scale in zip(
            tilts, defocus, bfactor, scale, strict=True
        ):
            output.write(
                f"{particle_defocus_u:.6f} 300.0 2.7 0.07 0.0 "
                f"{angle:.6f} 0.0 {dose_b:.6f} {tilt_scale:.8f}\n"
            )


def write_particles_star(
    path: Path,
    candidates: np.ndarray,
    tomogram_relative: Path,
    extract_root: Path,
    ctf_root: Path,
    image_pixel_size: float,
    image_size: int,
) -> None:
    with path.open("w") as output:
        output.write(
            "data_optics\n\n"
            "loop_\n"
            "_rlnOpticsGroupName #1\n"
            "_rlnOpticsGroup #2\n"
            "_rlnVoltage #3\n"
            "_rlnSphericalAberration #4\n"
            "_rlnAmplitudeContrast #5\n"
            "_rlnImagePixelSize #6\n"
            "_rlnImageSize #7\n"
            "_rlnImageDimensionality #8\n"
            "_rlnCtfDataAreCtfPremultiplied #9\n"
            f"empiar10045 1 300.0 2.7 0.07 {image_pixel_size:.9f} "
            f"{image_size} 3 0\n\n"
            "data_particles\n\n"
            "loop_\n"
            "_rlnMicrographName #1\n"
            "_rlnCoordinateX #2\n"
            "_rlnCoordinateY #3\n"
            "_rlnCoordinateZ #4\n"
            "_rlnImageName #5\n"
            "_rlnCtfImage #6\n"
            "_rlnOpticsGroup #7\n"
            "_rlnAutopickFigureOfMerit #8\n"
        )
        for index, (z, y, x, score) in enumerate(candidates, start=1):
            suffix = f"{index:06d}"
            particle = extract_root / f"{TOMOGRAM_NAME}{suffix}.mrc"
            ctf = ctf_root / f"{TOMOGRAM_NAME}_ctf{suffix}.mrc"
            output.write(
                f"{tomogram_relative} {x:.6f} {y:.6f} {z:.6f} "
                f"{particle} {ctf} 1 {score:.9g}\n"
            )


def write_run_scripts(
    output_dir: Path,
    box_size: int,
    scaled_size: int,
    pixel_size: float,
) -> None:
    scaled_pixel_size = pixel_size * box_size / scaled_size
    extraction = output_dir / "run_extract_particles.sh"
    extraction.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'RELION_BIN_DIR="${RELION_BIN_DIR:-/opt/relion4/bin}"\n'
        'cd "$(dirname "$0")"\n'
        '"${RELION_BIN_DIR}/relion_preprocess" \\\n'
        "  --i all_tomograms.star \\\n"
        "  --coord_suffix _top10n.star \\\n"
        "  --coord_dir ./ \\\n"
        "  --part_dir Extract/top10n/ \\\n"
        "  --part_star extracted_particles.star \\\n"
        "  --extract \\\n"
        f"  --extract_size {box_size} \\\n"
        f"  --scale {scaled_size} \\\n"
        "  --norm \\\n"
        f"  --bg_radius {int(0.75 * scaled_size / 2)} \\\n"
        "  --invert_contrast\n"
    )
    extraction.chmod(0o755)

    reconstruct = output_dir / "run_reconstruct_ctfs.sh"
    reconstruct.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'RELION_BIN_DIR="${RELION_BIN_DIR:-/opt/relion4/bin}"\n'
        'JOBS="${JOBS:-4}"\n'
        "export RELION_BIN_DIR\n"
        'cd "$(dirname "$0")"\n'
        "find Particles/Tomograms/08 -name '*_ctf??????.star' -print0 | "
        "sort -z | xargs -0 -r -n 1 -P \"${JOBS}\" bash -c '\n"
        "  set -euo pipefail\n"
        "  input=\"$1\"\n"
        "  output=\"${input%.star}.mrc\"\n"
        f"  temporary=\"${{input%.star}}.full{box_size}.mrc\"\n"
        "  if [[ -s \"$output\" ]]; then exit 0; fi\n"
        "  \"${RELION_BIN_DIR}/relion_reconstruct\" --i \"$input\" --o \"$temporary\" "
        f"--reconstruct_ctf {box_size} --angpix {pixel_size:.9f}\n"
        "  \"${RELION_BIN_DIR}/relion_image_handler\" --i \"$temporary\" --o \"$output\" "
        f"--new_box {scaled_size} "
        f"--force_header_angpix {scaled_pixel_size:.9f}\n"
        "  rm -f -- \"$temporary\"\n"
        "' _\n"
    )
    reconstruct.chmod(0o755)


def main() -> None:
    args = parse_args()
    if args.box_size <= 0 or args.box_size % 2:
        raise ValueError("box size must be a positive even integer")
    if args.scaled_size <= 0 or args.scaled_size % 2:
        raise ValueError("scaled size must be a positive even integer")
    if args.scaled_size > args.box_size:
        raise ValueError("scaled size cannot exceed extraction box size")

    candidates = np.asarray(np.load(require_file(args.candidates)), dtype=np.float64)
    if candidates.ndim != 2 or candidates.shape[1] != 4:
        raise ValueError("candidates must have (z, y, x, score) columns")
    if not np.all(np.isfinite(candidates)):
        raise ValueError("candidates contain non-finite values")
    if np.any(np.diff(candidates[:, 3]) > 0):
        raise ValueError("candidates are not sorted by descending score")
    if args.max_picks is not None:
        if args.max_picks < 1:
            raise ValueError("max picks must be positive")
        candidates = candidates[: args.max_picks]

    source_tomogram_dir = args.dataset_root / "Tomograms" / "08"
    source_anticipated = args.dataset_root / "AnticipatedResults"
    source_base = source_tomogram_dir / TOMOGRAM_NAME
    source_files = {
        suffix: require_file(source_base.with_suffix(suffix))
        for suffix in (".mrc", ".mrcs", ".order", ".tlt", ".trial")
    }
    source_ctf = require_file(
        source_anticipated
        / "Tomograms"
        / "08"
        / "ctffind"
        / f"{TOMOGRAM_NAME}_ctffind.star"
    )

    with mrcfile.open(source_files[".mrc"], permissive=True, header_only=True) as mrc:
        tomogram_zyx = (int(mrc.header.nz), int(mrc.header.ny), int(mrc.header.nx))
        header_pixel_size = float(mrc.voxel_size.x)
    with mrcfile.open(source_files[".mrcs"], permissive=True, header_only=True) as mrc:
        tilt_stack_zyx = (int(mrc.header.nz), int(mrc.header.ny), int(mrc.header.nx))

    coordinates = candidates[:, :3]
    if np.any(coordinates < 0) or np.any(coordinates >= np.asarray(tomogram_zyx)):
        raise ValueError("candidate coordinate lies outside the tomogram")
    half_box = args.box_size // 2
    full_box = np.all(
        (coordinates >= half_box)
        & (coordinates < np.asarray(tomogram_zyx) - half_box),
        axis=1,
    )

    tilts = np.atleast_1d(np.loadtxt(source_files[".tlt"], dtype=np.float64))
    order = np.atleast_2d(np.loadtxt(source_files[".order"], dtype=np.float64))
    if tilt_stack_zyx[0] != tilts.size:
        raise ValueError(
            f"tilt stack has {tilt_stack_zyx[0]} images but .tlt has {tilts.size} rows"
        )
    base_defocus = deposited_defocus_per_tilt(
        source_ctf, tilts, args.low_tilt_limit
    )
    dose_bfactor = matched_doses(tilts, order) * args.dose_bfactor

    output_dir = args.output_dir.resolve()
    tomo_dir = output_dir / "Tomograms" / "08"
    ctf_dir = output_dir / "Particles" / "Tomograms" / "08"
    extract_dir = output_dir / "Extract" / "top10n" / "Tomograms" / "08"
    for directory in (tomo_dir, ctf_dir, extract_dir):
        directory.mkdir(parents=True, exist_ok=True)

    for suffix, source in source_files.items():
        replace_symlink(tomo_dir / f"{TOMOGRAM_NAME}{suffix}", source)

    coordinate_star = tomo_dir / f"{TOMOGRAM_NAME}_top10n.star"
    write_coordinate_star(coordinate_star, candidates)
    write_coordinate_text(tomo_dir / f"{TOMOGRAM_NAME}.coords", candidates)

    tomogram_relative = Path("Tomograms/08") / f"{TOMOGRAM_NAME}.mrc"
    (output_dir / "all_tomograms.star").write_text(
        "data_optics\n\n"
        "loop_\n"
        "_rlnOpticsGroupName #1\n"
        "_rlnOpticsGroup #2\n"
        "_rlnMicrographPixelSize #3\n"
        "_rlnVoltage #4\n"
        "_rlnSphericalAberration #5\n"
        "_rlnAmplitudeContrast #6\n"
        f"empiar10045 1 {args.legacy_pixel_size:.9f} 300.0 2.7 0.07\n\n"
        "data_micrographs\n\n"
        "loop_\n"
        "_rlnMicrographName #1\n"
        "_rlnOpticsGroup #2\n"
        f"{tomogram_relative} 1\n"
    )

    xyz = candidates[:, [2, 1, 0]]
    per_particle_defocus = particle_defocus(
        xyz,
        tilts,
        base_defocus,
        (tomogram_zyx[2], tomogram_zyx[1], tomogram_zyx[0]),
        args.legacy_pixel_size,
    )
    for index, defocus in enumerate(per_particle_defocus, start=1):
        write_particle_ctf_star(
            ctf_dir / f"{TOMOGRAM_NAME}_ctf{index:06d}.star",
            tilts,
            defocus,
            dose_bfactor,
        )

    write_particles_star(
        output_dir / "particles_subtomo.star",
        candidates,
        tomogram_relative,
        Path("Extract/top10n/Tomograms/08"),
        Path("Particles/Tomograms/08"),
        args.legacy_pixel_size * args.box_size / args.scaled_size,
        args.scaled_size,
    )
    write_run_scripts(
        output_dir,
        args.box_size,
        args.scaled_size,
        args.legacy_pixel_size,
    )

    particle_bytes = candidates.shape[0] * args.scaled_size**3 * 4
    ctf_bytes = candidates.shape[0] * args.scaled_size**3 * 4
    temporary_ctf_bytes = args.box_size**3 * 4
    manifest = {
        "candidate_source": str(args.candidates.resolve()),
        "candidate_count": int(candidates.shape[0]),
        "candidate_columns": ["z", "y", "x", "score"],
        "relion_coordinate_order": ["x", "y", "z"],
        "complete_unpadded_box_count": int(np.sum(full_box)),
        "boundary_padded_box_count": int(np.sum(~full_box)),
        "tomogram_shape_zyx": list(tomogram_zyx),
        "tilt_stack_shape_zyx": list(tilt_stack_zyx),
        "tilt_count": int(tilts.size),
        "box_size": args.box_size,
        "scaled_size": args.scaled_size,
        "legacy_pixel_size_angstrom": args.legacy_pixel_size,
        "scaled_pixel_size_angstrom": (
            args.legacy_pixel_size * args.box_size / args.scaled_size
        ),
        "tomogram_header_pixel_size_angstrom": header_pixel_size,
        "particle_diameter_angstrom": args.particle_diameter,
        "low_tilt_limit_degrees": args.low_tilt_limit,
        "dose_bfactor_per_electron_per_angstrom2": args.dose_bfactor,
        "estimated_scaled_particle_bytes": particle_bytes,
        "estimated_scaled_ctf_bytes": ctf_bytes,
        "estimated_temporary_ctf_bytes_per_worker": temporary_ctf_bytes,
        "note": (
            "Each CTF is reconstructed at the extraction box size and immediately "
            "rewindowed to the scaled classification box size."
        ),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Prepared {candidates.shape[0]:,} ranked picks in {output_dir}")
    print(
        f"Complete {args.box_size}^3 boxes: {np.sum(full_box):,}; "
        f"boundary-padded boxes: {np.sum(~full_box):,}"
    )
    print(
        f"Scaled particle storage estimate: {particle_bytes / 2**30:.1f} GiB; "
        f"scaled CTF estimate: {ctf_bytes / 2**30:.1f} GiB"
    )
    print("Metadata preparation only; no extraction or reconstruction was launched.")


if __name__ == "__main__":
    main()
