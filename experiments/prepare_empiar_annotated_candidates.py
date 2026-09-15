#!/usr/bin/env python3
"""Convert deposited EMPIAR-10045 XYZ annotations into ranked ZYX-score arrays."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tomograms", nargs="+", default=["05", "08", "09", "10"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for raw_id in args.tomograms:
        tomo_id = f"{int(raw_id):02d}"
        name = f"IS002_291013_{int(raw_id):03d}"
        source = (
            args.dataset_root
            / "AnticipatedResults"
            / "Tomograms"
            / tomo_id
            / f"{name}.coords"
        )
        xyz = np.loadtxt(source, dtype=np.float64, ndmin=2)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"{source}: expected XYZ coordinate rows")
        # The RELION preparation helper expects descending-score (z,y,x,score).
        candidates = np.column_stack((xyz[:, ::-1], np.zeros(xyz.shape[0])))
        destination = args.output_dir / f"tomo{tomo_id}_annotated_zyx_score.npy"
        np.save(destination, candidates)
        print(f"{tomo_id}: {len(candidates)} annotations -> {destination}")
        total += len(candidates)
    print(f"Total annotations: {total}")


if __name__ == "__main__":
    main()
