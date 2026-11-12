"""Quick checks for the replication code. They take about ten seconds.

Run with either of:
    python3 tests/test_replication.py
    python3 -m pytest tests

No data is downloaded. The model tests use the synthetic demo inputs or the
saved 2021 inputs in outputs/benchmark_1d.
"""
from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import before_factory_opens_2d_geographic_extension as two_d  # noqa: E402
import before_factory_opens_main_model as core  # noqa: E402
import before_factory_opens_robustness as robustness  # noqa: E402

DEMO = replace(core.Calibration(), nx=31, dt=0.20, tail_years=4.0)


def test_news_dates_round_up_to_the_grid():
    assert abs(core.grid_at_or_after(2.16, 0.10) - 2.2) < 1e-12
    assert abs(core.grid_at_or_after(4.0, 0.10) - 4.0) < 1e-12   # already on the grid: stays put
    assert abs(core.grid_at_or_after(4.0, 0.30) - 4.2) < 1e-12


def test_band_weights_measure_the_band_exactly():
    m, *_ = core.build_model(DEMO, *core.demo_inputs(DEMO))
    g = core.grid(m)
    width = float(core.interval_weights(g, m.band_h) @ np.full(g.n, g.dx))
    assert abs(width - 2 * m.band_h) < 1e-12


def test_generator_matches_a_cell_by_cell_build():
    m, *_ = core.build_model(DEMO, *core.demo_inputs(DEMO))
    g = core.grid(m)
    a = np.random.default_rng(0).random((m.nx, 2))
    dense = np.zeros((m.nx, m.nx))
    r = 0.5 * m.sigma2 / g.dx ** 2
    for i in range(m.nx):
        if i > 0:
            dense[i, i - 1] = r + a[i, 0] / g.dx
        if i < m.nx - 1:
            dense[i, i + 1] = r + a[i, 1] / g.dx
        dense[i, i] = -dense[i].sum()
    assert np.allclose(core.generator(a, g, m).toarray(), dense, rtol=0, atol=1e-10)


def test_baseline_is_an_exact_equilibrium_and_stays_put():
    raw, qcew = core.demo_inputs(DEMO)
    m, g, p0, pre, post, known, surprise, diag = core.solve_all(DEMO, raw, qcew)
    assert diag["baseline_target_vs_stationary_tv"] < 1e-12
    assert diag["no_project_tv_at_opening"] < 1e-12
    # Before opening, the surprise economy has not moved; the forward-looking one has.
    i = core.time_index(g, m.opening)
    assert core.tv(surprise.mass[i], p0) < 1e-12
    assert core.tv(known.mass[i], p0) > 1e-4
    assert abs(known.mass.sum(1) - 1).max() < 1e-12


def test_replay_reproduces_the_saved_headline_numbers():
    z = json.loads((core.REFERENCE / "Benchmark_1D_Calibration.json").read_text())
    saved = json.loads((core.REFERENCE / "Benchmark_1D_Headline_Metrics.json").read_text())
    cal = core.Calibration(**z["calibration"])
    m, g, p0, pre, post, known, surprise, _ = core.solve_all(cal, np.asarray(z["data"]["raw_mass"]), z["data"]["qcew"])
    new = core.headline(m, g, pre, post, known, surprise)
    assert max(abs(saved[k] - new[k]) for k, v in saved.items() if isinstance(v, float)) < 1e-9


def test_two_d_startup_checks_pass():
    assert two_d.unit_checks()["all_passed"]


def test_time_step_extrapolation_recovers_a_known_limit():
    import pandas as pd
    # r(dt) = 0.5 - 0.3 * dt: first-order convergence to 0.5.
    d = pd.DataFrame({"status": "ok", "dt": [0.20, 0.10, 0.05]})
    d["tv_ratio"] = 0.5 - 0.3 * d.dt
    est = robustness.time_step_extrapolation(d)
    assert abs(est["observed_order"] - 1.0) < 1e-9
    assert abs(est["tv_ratio_limit"] - 0.5) < 1e-9


def test_output_folders_are_only_replaced_when_safe():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "notes.txt").write_text("not ours")
        for unsafe in (tmp, core.REFERENCE):
            try:
                core.fresh_output_dir(unsafe, "Benchmark_1D_Calibration.json")
            except SystemExit:
                pass
            else:
                raise AssertionError(f"{unsafe} should have been refused")
        assert (tmp / "notes.txt").exists()

        earlier = tmp / "earlier_run"
        earlier.mkdir()
        (earlier / "Benchmark_1D_Calibration.json").write_text("{}")
        core.fresh_output_dir(earlier, "Benchmark_1D_Calibration.json")
        assert earlier.is_dir() and not any(earlier.iterdir())


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} checks passed")
