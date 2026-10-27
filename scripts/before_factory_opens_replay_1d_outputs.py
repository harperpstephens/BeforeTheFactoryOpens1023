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
import pandas as pd

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

    m, g, p0, pre, post, known, surprise, diag = core.solve_all(cal, raw, qcew)

    # Sequential revisions, on the same news dates and expected openings as the live run.
    r1 = core.grid_at_or_after(core.years(cal.announcement, cal.revision1_date), m.dt)
    o1 = core.grid_at_or_after(cal.revision1_opening_lag, m.dt)
    r2 = core.grid_at_or_after(core.years(cal.announcement, cal.revision2_date), m.dt)
    o2 = core.grid_at_or_after(cal.revision2_opening_lag, m.dt)
    rm, details = core.revision_path(
        m, g, known, post,
        [("2024_revision", r1, o1), ("2025_revision", r2, o2)],
    )
    i0 = core.time_index(g, m.opening)
    rev = {
        "events": details,
        "max_mass_error": float(np.max(np.abs(rm.sum(1) - 1))),
        "min_mass": float(rm.min()),
        "gap_at_initial_expected_opening_pp": 100 * (
            core.metrics(known.mass[i0], g, m)["band_mass"]
            - core.metrics(rm[i0], g, m)["band_mass"]
        ),
        "tv_at_initial_expected_opening": core.tv(known.mass[i0], rm[i0]),
        "note": "Offline replay of preserved live public inputs; no synthetic data substituted.",
    }
    rev["all_passed"] = bool(
        rev["max_mass_error"] < 1e-10
        and rev["min_mass"] >= -1e-12
        and all(
            e["inheritance_error"] < 1e-12
            and e["fp"] < 2e-6
            and e["hjb"] < 1e-7
            and e["kfe"] < 1e-10
            and e["min_mass"] >= -1e-12
            for e in details
        )
    )
    if not rev["all_passed"]:
        raise RuntimeError(f"Revision-path checks failed: {rev}")

    head, val = core.write_results(out, cal, m, g, p0, pre, post, known, surprise, diag, meta, rev)
    pd.DataFrame(
        [{"time": float(t), **core.metrics(rm[i], g, m)} for i, t in enumerate(g.t)]
    ).to_csv(out / "Benchmark_1D_Sequential_Revision_Path.csv", index=False)
    print(json.dumps({"version": core.VERSION, "headline": head, "validation": val}, indent=2))


if __name__ == "__main__":
    main()
