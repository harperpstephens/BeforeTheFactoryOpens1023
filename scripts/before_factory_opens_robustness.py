"""Before the Factory Opens: robustness and numerical-refinement suite (r11).

Re-solves the one-dimensional benchmark under alternative parameters, grids,
and timing assumptions, and saves one CSV per group of checks.

Every row is either a solved specification or an explicit numerical failure
with its error message. Failed cases are never turned into zero effects and
never dropped from the tables.

Besides the per-suite CSVs, the run writes Robustness_Summary.csv (successes
and failures per suite), Table_C1_Robustness_Summary.csv (the ranges in the
paper's Table C1), and Robustness_Report.json, which also holds a
time-step extrapolation of the headline ratio.

Usage (from the repository root):
    python3 scripts/before_factory_opens_robustness.py
    python3 scripts/before_factory_opens_robustness.py --smoke
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import before_factory_opens_main_model as core


def load_inputs(root: Path):
    """Read the calibration and data saved by a benchmark (or replay) run."""
    z = json.loads((root / "Benchmark_1D_Calibration.json").read_text())
    cal = core.Calibration(**z["calibration"])
    d = z["data"]
    return cal, np.asarray(d["raw_mass"], float), d["qcew"], d.get("data_mode", "")


def run_case(cal, raw, q, label, **kw):
    """Solve one specification and return a table row, or the error if it fails."""
    row = {"case": label, "status": "failed"}
    try:
        m, g, p0, pre, post, known, surprise, diag = core.solve_all(cal, raw, q, **kw)
        h = core.headline(m, g, pre, post, known, surprise)
        row.update(
            status="ok",
            opening_tv=h["opening_tv"],
            tv_ratio=h["tv_anticipation_ratio"],
            projected_band_gap_pp=h["projected_band_gap_pp"],
            band_ratio=h["band_anticipation_ratio"],
            no_project_tv_at_opening=diag["no_project_tv_at_opening"],
            baseline_target_tv=diag["baseline_target_vs_stationary_tv"],
            pre_kfe=pre.kfe,
            post_kfe=post.kfe,
            known_fp=known.fp,
            surprise_fp=surprise.fp,
        )
    except Exception as e:
        row["error"] = str(e)
    row.update(kw)
    return row


# Folder and file name for each group of checks.
SUITE_OUTPUT_NAMES = {
    "structural": ("01_Structural_Parameters", "Structural_Parameters.csv"),
    "smoothing": ("02_Smoothing_Bandwidth", "Smoothing_Bandwidth.csv"),
    "project_shock": ("03_Project_Shock", "Project_Shock_Amplitude_and_Width.csv"),
    "grid": ("04_Spatial_Grid", "Spatial_Grid_Refinement.csv"),
    "time_step": ("05_Time_Step", "Time_Step_Refinement.csv"),
    "terminal_tail": ("06_Terminal_Tail", "Terminal_Tail_Length.csv"),
    "opening_expectation": ("07_Opening_Horizon", "Opening_Horizon_Sensitivity.csv"),
    "revision_timing": ("08_Sequential_Revisions", "Sequential_Revision_Horizons.csv"),
    "stress_nonconvergent": ("09_Optional_Stress_Tests", "Optional_Stress_Tests.csv"),
    "descriptive_lodes": ("10_Descriptive_LODES_Checks", "Descriptive_LODES_Checks.csv"),
    "smoke": ("00_Smoke_Test", "Smoke_Test.csv"),
}


# Suites kept out of the reported ranges: the optional stress cases and the
# descriptive LODES table (no model is solved there).
UNREPORTED = {"stress_nonconvergent", "descriptive_lodes"}

# Rows of the paper's Table C1, in order: (label, suite, column holding opening TV).
TABLE_C1 = [
    ("Structural parameters", "structural", "opening_tv"),
    ("Smoothing bandwidth", "smoothing", "opening_tv"),
    ("Project amplitude and width", "project_shock", "opening_tv"),
    ("Spatial grid", "grid", "opening_tv"),
    ("Time step", "time_step", "opening_tv"),
    ("Terminal tail", "terminal_tail", "opening_tv"),
    ("Initial opening horizon", "opening_expectation", "opening_tv"),
    ("Sequential revision horizons", "revision_timing", "tv_at_initial_expected_opening"),
]


def save_suite(name, cases, out):
    """Write one group of results to its own folder and return it as a DataFrame."""
    df = pd.DataFrame(cases)
    folder, filename = SUITE_OUTPUT_NAMES.get(name, (name, f"{name}.csv"))
    d = out / folder
    d.mkdir(parents=True, exist_ok=True)
    df.to_csv(d / filename, index=False)
    return df


def structural(cal, raw, q, out, smoke=False):
    """Moving cost c, idiosyncratic noise sigma2, discount rate rho, congestion gamma and kappa."""
    # The default suite holds the reference case and every perturbation that
    # converged within the r11 validation budget. Harder extremes live in
    # stress(), where they are reported, rather than dropped or tuned until
    # they happen to converge.
    specs = [("reference", {}), ("c=1.0", {"c": 1.0}), ("c=2.0", {"c": 2.0}),
             ("sigma2=.20", {"sigma2": .20}), ("rho=.03", {"rho": .03}), ("rho=.07", {"rho": .07}),
             ("gamma=1.5", {"gamma": 1.5}), ("kappa=.5x", {"kappa_mult": .5})]
    if smoke:
        specs = specs[:2]
    return save_suite("structural", [run_case(cal, raw, q, n, **k) for n, k in specs], out)


def stress(cal, raw, q, out):
    """Optional cases that are numerically hard; run only with --stress."""
    specs = [("sigma2=.10", {"sigma2": .10}), ("gamma=.5", {"gamma": .5}), ("kappa=2x", {"kappa_mult": 2.0})]
    return save_suite("stress_nonconvergent", [run_case(cal, raw, q, n, **k) for n, k in specs], out)


def smoothing(cal, raw, q, out, smoke=False):
    """Bandwidth (in miles) used to smooth the LODES residence pattern."""
    vals = [2.0, 3.5, 5.0, 7.5] if not smoke else [3.5, 5.0]
    return save_suite("smoothing", [run_case(cal, raw, q, f"smooth={v}", smoothing=v) for v in vals], out)


def shock(cal, raw, q, out, smoke=False):
    """Size of Intel's premium (multiplier on omega) and how far it reaches (miles)."""
    amps = [.5, 1, 1.5] if not smoke else [1.0]
    radii = [10., 20., 30.] if not smoke else [20., 30.]
    return save_suite("project_shock", [run_case(cal, raw, q, f"amp={a},h={r}", omega_mult=a, shock_scale_miles=r)
                                        for a in amps for r in radii], out)


def numerical(cal, raw, q, out, smoke=False):
    """Discretization checks: number of cells, time step, and length of the terminal tail."""
    grids = [41, 61, 81, 101] if not smoke else [31, 41]
    dts = [.05, .10, .20] if not smoke else [.20]
    tails = [8., 12., 16.] if not smoke else [4., 8.]
    a = save_suite("grid", [run_case(cal, raw, q, f"nx={v}", nx=v) for v in grids], out)

    # Hold the opening date and terminal horizon fixed across time steps.
    # Otherwise a coarser dt would also move the opening date on the grid,
    # mixing a numerical change with an economic one.
    common_open = core.grid_at_or_after(cal.initial_opening_lag, max(dts))
    latest = max(cal.initial_opening_lag, cal.revision1_opening_lag, cal.revision2_opening_lag)
    common_horizon = core.grid_at_or_after(latest + cal.tail_years, max(dts))
    common_tail = common_horizon - latest
    b = save_suite("time_step", [run_case(cal, raw, q, f"dt={v}", dt=v, opening=common_open, tail=common_tail)
                                 for v in dts], out)
    c = save_suite("terminal_tail", [run_case(cal, raw, q, f"tail={v}", tail=v) for v in tails], out)
    return a, b, c


def timing(cal, raw, q, out, smoke=False):
    """Alternative expected opening dates, in years after the announcement."""
    # These are assumptions about when workers expected the plant to open,
    # not attempts to pin down a single true calendar day.
    lags = [3.5, 4.0, 4.5] if not smoke else [4.0, 4.5]
    return save_suite("opening_expectation", [run_case(cal, raw, q, f"opening_lag={v}", opening=v) for v in lags], out)


def revision_timing(cal, raw, q, out):
    """Shift the revised opening dates while keeping the news dates fixed."""
    rows = []
    try:
        m, g, p0, pre, post, known, _, _ = core.solve_all(cal, raw, q)
        r1 = core.grid_at_or_after(core.years(cal.announcement, cal.revision1_date), m.dt)
        r2 = core.grid_at_or_after(core.years(cal.announcement, cal.revision2_date), m.dt)
        i0 = core.time_index(g, m.opening)
        for label, o1, o2 in [("earlier_horizons", 5.5, 8.5), ("reference", 6.0, 9.0), ("later_horizons", 6.5, 9.5)]:
            mass, events = core.revision_path(
                m, g, known, post,
                [("2024_revision", r1, core.grid_at_or_after(o1, m.dt)),
                 ("2025_revision", r2, core.grid_at_or_after(o2, m.dt))],
            )
            rows.append({
                "case": label,
                "status": "ok",
                "revision1_opening_lag": o1,
                "revision2_opening_lag": o2,
                "gap_at_initial_expected_opening_pp": 100 * (core.metrics(known.mass[i0], g, m)["band_mass"]
                                                             - core.metrics(mass[i0], g, m)["band_mass"]),
                "tv_at_initial_expected_opening": core.tv(known.mass[i0], mass[i0]),
                "max_mass_error": float(np.max(np.abs(mass.sum(1) - 1))),
                "max_segment_fp": max(e["fp"] for e in events),
            })
    except Exception as e:
        rows.append({"case": "suite_setup", "status": "failed", "error": str(e)})
    return save_suite("revision_timing", rows, out)


def holdout(cal, out, years=(2021, 2022, 2023)):
    """Descriptive only: share of LODES residents near the site in 2021-2023.

    No model is solved here. It needs a live download and runs only with --holdout.
    """
    rows = []
    for y in years:
        try:
            p, n, geo5 = core.lodes_mass(cal, y)
            dx = 2 / cal.nx
            x = -1 + (np.arange(cal.nx) + .5) * dx
            g = core.Grid(x, dx, np.array([0.0]))
            band = float(p @ core.interval_weights(g, cal.central_band_miles / cal.half_length_miles))
            rows.append({"year": y, "status": "ok", "workers": n, "projected_band_share": band,
                         "geographic_5mi_share": geo5})
        except Exception as e:
            rows.append({"year": y, "status": "failed", "error": str(e)})
    return save_suite("descriptive_lodes", rows, out)


def time_step_extrapolation(d: pd.DataFrame):
    """Estimate the headline ratio as the time step goes to zero.

    The time-step suite solves the model at dt, dt/2, and dt/4. The two
    successive changes in the ratio give the observed order of convergence p,
    and Richardson extrapolation projects the finest run forward:

        limit = r(dt/4) + (r(dt/4) - r(dt/2)) / (2**p - 1)

    This is an estimate from three runs, not a solved specification. Returns
    None unless all three runs solved and the ratio moves monotonically.
    """
    d = d[d.status == "ok"].sort_values("dt", ascending=False)
    if len(d) != 3 or not np.allclose(d.dt.to_numpy()[:-1] / d.dt.to_numpy()[1:], 2.0):
        return None
    coarse, mid, fine = (float(v) for v in d.tv_ratio)
    if (mid - coarse) * (fine - mid) <= 0:
        return None
    order = math.log2((mid - coarse) / (fine - mid))
    return {
        "time_steps": [float(v) for v in d.dt],
        "tv_ratio": [coarse, mid, fine],
        "observed_order": order,
        "tv_ratio_limit": fine + (fine - mid) / (2 ** order - 1),
        "tv_ratio_limit_if_first_order": 2 * fine - mid,
        "note": "Richardson extrapolation from three time steps; an estimate, not a solved specification.",
    }


def summary(out: Path):
    """Count successes and failures per suite and write the ranges the paper reports.

    The ranges cover only specifications that solved, and leave out the
    optional stress cases. They describe how much the result moves across
    checks; they are not confidence intervals.
    """
    suites = {}
    for name, (folder, filename) in SUITE_OUTPUT_NAMES.items():
        if (out / folder / filename).exists():
            suites[name] = pd.read_csv(out / folder / filename)

    rows = [{"suite": Path(SUITE_OUTPUT_NAMES[name][1]).stem, "attempted": len(d),
             "successful": int((d.status == "ok").sum()), "failed": int((d.status != "ok").sum())}
            for name, d in suites.items()]
    pd.DataFrame(rows).to_csv(out / "Robustness_Summary.csv", index=False)

    reported = {name: d[d.status == "ok"] for name, d in suites.items() if name not in UNREPORTED}
    good = pd.concat(reported.values(), ignore_index=True) if reported else pd.DataFrame()

    report = {"suites": rows, "note": "Ranges below condition on numerical success; they are not confidence intervals."}
    if len(good) and "tv_ratio" in good:
        v = pd.to_numeric(good.tv_ratio, errors="coerce").dropna()
        b = pd.to_numeric(good.band_ratio, errors="coerce").dropna()
        report.update(
            tv_ratio_min=float(v.min()) if len(v) else None,
            tv_ratio_max=float(v.max()) if len(v) else None,
            band_ratio_min=float(b.min()) if len(b) else None,
            band_ratio_max=float(b.max()) if len(b) else None,
        )
    if "time_step" in suites:
        report["time_step_extrapolation"] = time_step_extrapolation(suites["time_step"])
    (out / "Robustness_Report.json").write_text(json.dumps(report, indent=2))

    # Table C1: one row per suite, with the range of opening TV and of the TV timing ratio.
    table = []
    for label, name, tv_column in TABLE_C1:
        if name not in suites:
            continue
        d, ok = suites[name], reported[name]
        row = {"variation": label, "successful_cases": f"{len(ok)}/{len(d)}"}
        if len(ok):
            row.update(opening_tv_min=ok[tv_column].min(), opening_tv_max=ok[tv_column].max())
            if "tv_ratio" in ok:
                row.update(tv_timing_pct_min=100 * ok.tv_ratio.min(), tv_timing_pct_max=100 * ok.tv_ratio.max())
        table.append(row)
    if table:
        pd.DataFrame(table).to_csv(out / "Table_C1_Robustness_Summary.csv", index=False)

    macros = []
    if report.get("tv_ratio_min") is not None:
        macros = [("TVRatioRobustMin", report["tv_ratio_min"], ".3f", 1),
                  ("TVRatioRobustMax", report["tv_ratio_max"], ".3f", 1),
                  ("TVRatioRobustMinPct", report["tv_ratio_min"], ".1f", 100),
                  ("TVRatioRobustMaxPct", report["tv_ratio_max"], ".1f", 100)]
    core.write_macros(out / "Manuscript_Robustness_Results.tex", "robustness suite (r11)", macros)
    return report


def main():
    ap = argparse.ArgumentParser(description="Robustness checks for the Before the Factory Opens 1D benchmark.")
    ap.add_argument("--pipeline-output", type=Path, default=core.REFERENCE,
                    help="Folder from a benchmark or replay run, holding Benchmark_1D_Calibration.json "
                         "(default: outputs/benchmark_1d).")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Where to write results (default: outputs/robustness, or outputs/robustness_smoke "
                         "with --smoke). An earlier run in that folder is replaced.")
    ap.add_argument("--smoke", action="store_true",
                    help="Run three quick cases on a coarse grid to check the code works.")
    ap.add_argument("--holdout", action="store_true",
                    help="Also download 2021-2023 LODES for a descriptive comparison.")
    ap.add_argument("--stress", action="store_true", help="Also try the optional cases that are known to be numerically hard.")
    a = ap.parse_args()

    cal, raw, q, mode = load_inputs(a.pipeline_output)
    out = core.fresh_output_dir(a.output_dir or core.OUTPUTS / ("robustness_smoke" if a.smoke else "robustness"),
                                "Robustness_Summary.csv", *(folder for folder, _ in SUITE_OUTPUT_NAMES.values()))

    if a.smoke:
        # Keep the smoke test small and numerically easy. The full run covers
        # every sensitivity group and records any non-convergence explicitly.
        base = {"nx": 21, "dt": .40, "tail": 4.0}
        cases = [run_case(cal, raw, q, "reference", **base),
                 run_case(cal, raw, q, "congestion_scale", kappa_mult=.5, **base),
                 run_case(cal, raw, q, "opening_expectation", opening=4.5, **base)]
        save_suite("smoke", cases, out)
    else:
        structural(cal, raw, q, out)
        smoothing(cal, raw, q, out)
        shock(cal, raw, q, out)
        numerical(cal, raw, q, out)
        timing(cal, raw, q, out)
        revision_timing(cal, raw, q, out)
        if a.stress:
            stress(cal, raw, q, out)
        if a.holdout and "synthetic" not in mode:
            holdout(cal, out)
    print(json.dumps(summary(out), indent=2))


if __name__ == "__main__":
    main()
