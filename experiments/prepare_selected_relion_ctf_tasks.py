"""Rewrite selected RELION particle CTF paths and emit reconstruction tasks."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-star", type=Path, required=True)
    parser.add_argument("--output-star", type=Path, required=True)
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--ctf-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lines = args.input_star.read_text().splitlines()
    labels: list[str] = []
    in_particles = False
    rewritten: list[str] = []
    tasks: list[str] = []
    particle_count = 0

    for raw_line in lines:
        stripped = raw_line.strip()
        if stripped == "data_particles":
            in_particles = True
            labels = []
            rewritten.append(raw_line)
            continue
        if in_particles and stripped.startswith("_rln"):
            labels.append(stripped.split()[0])
            rewritten.append(raw_line)
            continue
        if in_particles and labels and stripped and not stripped.startswith("#"):
            fields = stripped.split()
            if len(fields) == len(labels):
                ctf_column = labels.index("_rlnCtfImage")
                old_mrc = Path(fields[ctf_column])
                old_star = old_mrc.with_suffix(".star")
                relative = old_mrc.relative_to("Particles")
                new_mrc = args.ctf_dir / relative
                new_mrc.parent.mkdir(parents=True, exist_ok=True)
                fields[ctf_column] = new_mrc.as_posix()
                tasks.append(f"{old_star.as_posix()}\t{new_mrc.as_posix()}")
                rewritten.append(" ".join(fields))
                particle_count += 1
                continue
        rewritten.append(raw_line)

    if particle_count == 0:
        raise RuntimeError("input STAR file contains no particle rows")
    args.output_star.write_text("\n".join(rewritten) + "\n")
    args.task_file.write_text("\n".join(tasks) + "\n")
    print(f"Prepared {particle_count} selected CTF reconstruction tasks")


if __name__ == "__main__":
    main()
