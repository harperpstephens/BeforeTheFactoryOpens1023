"""Before the Factory Opens: rerun the 1D benchmark offline from saved inputs.

Reads the 2021 LODES residence shares and the QCEW summary that a live run of
before_factory_opens_main_model.py saved, then solves the same r11 model again.
Nothing is downloaded, and no synthetic data is substituted.

Usage:
    python before_factory_opens_replay_1d_outputs.py --preserved-output outputs/benchmark_1d \
        --output-dir outputs/replayed_1d_results
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

import before_factory_opens_main_model as core


def main() -> None:
    ap = argparse.ArgumentParser(description="Rerun the 1D benchmark from saved inputs, without downloading data.")
    ap.add_argument("--preserved-output", type=Path, required=True,
                    help="Folder from a live benchmark run (must contain Benchmark_1D_Calibration.json).")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Where to write results. An existing folder is replaced.")
    args = ap.parse_args()

    z = json.loads((args.preserved_output / "Benchmark_1D_Calibration.json").read_text(encoding="utf-8"))
    cal = core.Calibration(**z["calibration"])
    data = z["data"]
    raw = np.asarray(data["raw_mass"], dtype=float)
    qcew = data["qcew"]
    meta = dict(data)
    meta["data_mode"] = "preserved_live_public_inputs_offline_replay"

    out = args.output_dir
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    core.run(cal, raw, qcew, meta, out, "Offline replay of preserved live public inputs; no synthetic data substituted.")


if __name__ == "__main__":
    main()
