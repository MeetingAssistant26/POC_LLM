#!/usr/bin/env python3
"""Prepare Docker-contained reduced livekit-wakeword data for initial training.

Run through `wakeword/scripts/docker-run.sh`; do not execute this on the host.
It creates `wakeword/data-initial` from already-downloaded production setup data:

- symlinks Piper/background/RIR assets to `wakeword/data`
- writes deterministic small ACAV and validation feature subsets
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

ROOT = Path("/workspace/wakeword")
SOURCE = ROOT / "data"
DEST = ROOT / "data-initial"
ACAV_ROWS = 2_000
VALIDATION_ROWS = 3_200  # 200 validation windows after reshape to (-1, 16, 96)


def ensure_symlink(name: str) -> None:
    link = DEST / name
    target = Path("..") / "data" / name
    if link.is_symlink() and os.readlink(link) == str(target):
        return
    if link.exists() or link.is_symlink():
        if link.is_dir() and not link.is_symlink():
            raise SystemExit(f"{link} exists and is not the expected symlink")
        link.unlink()
    link.symlink_to(target)


def save_subset(source: Path, destination: Path, rows: int) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    if not source.exists():
        raise SystemExit(f"Required source feature file is missing: {source}")
    data = np.load(source, mmap_mode="r")
    np.save(destination, np.asarray(data[:rows], dtype=np.float32))


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for name in ["piper", "backgrounds", "rirs"]:
        ensure_symlink(name)

    features = DEST / "features"
    features.mkdir(parents=True, exist_ok=True)

    acav_name = "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
    validation_name = "validation_set_features.npy"
    save_subset(SOURCE / "features" / acav_name, features / acav_name, ACAV_ROWS)
    save_subset(SOURCE / "features" / validation_name, features / validation_name, VALIDATION_ROWS)

    for path in [
        features / acav_name,
        features / validation_name,
        DEST / "piper",
        DEST / "backgrounds",
        DEST / "rirs",
    ]:
        detail = os.readlink(path) if path.is_symlink() else f"{path.stat().st_size} bytes"
        print(f"{path} -> {detail}")


if __name__ == "__main__":
    main()
