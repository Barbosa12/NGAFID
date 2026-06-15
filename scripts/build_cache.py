"""Build ``data/X_raw_L2048.npy`` from ``data/flight_data.pkl``.

The analysis notebooks (01-05) expect a fixed-length tensor

    X_raw  shape = (11446, 23, 2048)  dtype = float32

The public NGAFID download (``2days.tar.gz`` on Zenodo,
https://zenodo.org/records/6624956) ships only the *variable-length* raw
flights in ``flight_data.pkl`` — a dict ``{master_index: ndarray(T, 23)}`` —
plus ``flight_header.csv`` and ``stats.csv``. This script reconstructs the
fixed-length cache the notebooks load:

    1. order flights by the rows of ``flight_header.csv`` (column "Master Index"),
    2. forward/backward-fill NaNs per sensor,
    3. cubic-spline resample each flight along time from T to 2048 steps,
    4. stack to (N, 23, 2048) and save to ``data/X_raw_L2048.npy``.

Run once, after extracting ``2days.tar.gz`` into ``data/``:

    python scripts/build_cache.py

Note: this regenerates the cache from the public raw data. Minor numerical
differences from any previously distributed cache are possible; re-run all
notebooks so the results are self-consistent.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

DATA = Path(__file__).resolve().parent.parent / "data"
LENGTH = 2048


def main() -> None:
    hdr = pd.read_csv(DATA / "flight_header.csv")
    stats = pd.read_csv(DATA / "stats.csv", index_col=0)
    sensors = [c for c in stats.columns if c not in ("timestep", "cluster")]
    n_sensors = len(sensors)

    print(f"Flights: {len(hdr):,} | sensors: {n_sensors} | target length: {LENGTH}")
    print("Loading flight_data.pkl (this is large, ~2-3 GB) ...")
    with open(DATA / "flight_data.pkl", "rb") as f:
        flight_data = pickle.load(f)

    x_new = np.linspace(0.0, 1.0, LENGTH)
    n = len(hdr)
    X = np.empty((n, n_sensors, LENGTH), dtype=np.float32)

    for i, midx in enumerate(hdr["Master Index"].to_numpy()):
        # tolerate int/str key types
        key = midx if midx in flight_data else (
            int(midx) if int(midx) in flight_data else str(midx))
        arr = np.asarray(flight_data[key], dtype=np.float64)  # (T, 23)

        # forward/backward fill NaNs per sensor column
        arr = pd.DataFrame(arr).ffill().bfill().to_numpy()
        if arr.shape[1] != n_sensors:
            raise ValueError(
                f"flight {key}: got {arr.shape[1]} sensors, expected {n_sensors}")

        t = arr.shape[0]
        if t >= 2:
            res = CubicSpline(np.linspace(0.0, 1.0, t), arr, axis=0)(x_new)
        else:
            res = np.repeat(arr, LENGTH, axis=0)
        X[i] = res.T.astype(np.float32)  # (23, 2048)

        if (i + 1) % 2000 == 0 or (i + 1) == n:
            print(f"  {i + 1:>6}/{n}")

    out = DATA / "X_raw_L2048.npy"
    np.save(out, X)
    print(f"Saved {out}  shape={X.shape}  dtype={X.dtype}  "
          f"({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
