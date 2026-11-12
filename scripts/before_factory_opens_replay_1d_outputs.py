"""Before the Factory Opens: rerun the 1D benchmark offline from saved inputs.

Reads the 2021 LODES residence shares and the QCEW summary saved in
outputs/benchmark_1d, solves the same model again, and compares the new
headline numbers with the saved ones. Nothing is downloaded, and no synthetic
data is substituted.

The comparison allows for floating-point noise: different NumPy, SciPy, or BLAS
builds move the headline numbers in about the twelfth decimal place. Anything
larger than --tolerance makes the script exit with an error.

Usage (from the repository root):
    python3 scripts/before_factory_opens_replay_1d_outputs.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import before_factory_opens_main_model as core


def headline_gap(saved: dict, new: dict) -> float:
    """Largest absolute difference between two sets of headline numbers."""
    return max(abs(saved[k] - new[k]) for k, v in saved.items() if isinstance(v, float))


def main() -> None:
    ap = argparse.ArgumentParser(description="Rerun the 1D benchmark from saved inputs, without downloading data.")
    ap.add_argument("--preserved-output", type=Path, default=core.REFERENCE,
                    help="Folder holding Benchmark_1D_Calibration.json (default: outputs/benchmark_1d).")
    ap.add_argument("--output-dir", type=Path, default=core.OUTPUTS / "replayed_1d_results",
                    help="Where to write results (default: outputs/replayed_1d_results). "
                         "An earlier replay in that folder is replaced.")
    ap.add_argument("--tolerance", type=float, default=1e-9,
                    help="Largest allowed gap between the saved and replayed headline numbers.")
    args = ap.parse_args()

    z = json.loads((args.preserved_output / "Benchmark_1D_Calibration.json").read_text(encoding="utf-8"))
    cal = core.Calibration(**z["calibration"])
    data = z["data"]
    raw = np.asarray(data["raw_mass"], dtype=float)
    qcew = data["qcew"]
    meta = dict(data)
    meta["data_mode"] = "preserved_live_public_inputs_offline_replay"

    out = core.fresh_output_dir(args.output_dir, "Benchmark_1D_Calibration.json")
    core.run(cal, raw, qcew, meta, out, "Offline replay of preserved live public inputs; no synthetic data substituted.")

    saved_path = args.preserved_output / "Benchmark_1D_Headline_Metrics.json"
    if saved_path.exists():
        saved = json.loads(saved_path.read_text(encoding="utf-8"))
        new = json.loads((out / "Benchmark_1D_Headline_Metrics.json").read_text(encoding="utf-8"))
        gap = headline_gap(saved, new)
        print(f"Largest gap from the saved headline numbers: {gap:.2e} (tolerance {args.tolerance:.0e})")
        if gap > args.tolerance:
            raise SystemExit("The replay does not reproduce the saved headline numbers.")


if __name__ == "__main__":
    main()
