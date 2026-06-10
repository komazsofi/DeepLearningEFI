#!/usr/bin/env python3

import laspy
import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------
# SETTINGS (EDIT THESE)
# ---------------------------
LAS_DIR = "/data/SR16_DL/laz_clip_norm_00/"
NPY_DIR = "/data/SR16_DL/output_npy"
ZQ95_CSV = "/data/SR16_DL/als_zq95.csv"

NORMALIZE_XY = True


# ---------------------------
# FUNCTIONS
# ---------------------------
def las_to_numpy(las_path, normalize_xy=True):
    """
    Convert LAS to Nx3 numpy array (float32)
    """
    las = laspy.read(las_path)

    # Stack coordinates
    pc = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)

    # Normalize XY (same as your R code)
    if normalize_xy:
        pc[:, 0] -= pc[:, 0].mean()
        pc[:, 1] -= pc[:, 1].mean()

    return pc


def compute_zq95(pc):
    """
    Compute 95th percentile of Z
    """
    return float(np.quantile(pc[:, 2], 0.95))


# ---------------------------
# MAIN WORKFLOW
# ---------------------------
def main():

    las_dir = Path(LAS_DIR)
    npy_dir = Path(NPY_DIR)
    npy_dir.mkdir(parents=True, exist_ok=True)

    las_files = sorted(las_dir.glob("*.laz"))

    results = []

    for las_file in las_files:

        plot_id = las_file.stem
        npy_file = npy_dir / f"{plot_id}.npy"

        # Skip if already processed (restart-safe)
        if npy_file.exists():
            print(f"Skipping {plot_id} (already exists)")
            continue

        # Convert
        pc = las_to_numpy(las_file, NORMALIZE_XY)

        # Save npy
        np.save(npy_file, pc)

        # Compute zq95
        zq95 = compute_zq95(pc)

        results.append({
            "plot_id": plot_id,
            "als_zq95": zq95
        })

        print(f"Processed {plot_id}: {pc.shape[0]} points")

    # Save zq95 CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(ZQ95_CSV, index=False)
        print(f"Saved zq95 -> {ZQ95_CSV}")
    else:
        print("No new files processed.")


# ---------------------------
# ENTRY POINT
# ---------------------------
if __name__ == "__main__":
    main()