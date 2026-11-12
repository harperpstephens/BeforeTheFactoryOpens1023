# Before the Factory Opens

Replication code for "[Before the Factory Opens: Anticipation and Spatial Adjustment to Announced and Delayed Local Shocks](https://dx.doi.org/10.2139/ssrn.7500178)" by Harper Stephens (working paper, 2026).

The paper uses a spatial mean field game, calibrated to 2021 Census LODES residences, to measure how much workers relocate in anticipation of Intel's Ohio semiconductor plant before it opens.

## Quick start

```bash
python3 -m pip install -r requirements.txt
```

```bash
./run_full_replication.sh
```

That runs the checks in `tests/`, re-solves the 1D model from the saved 2021 inputs, runs the robustness suite, and draws Figures 1–3. It takes about two minutes and downloads nothing. The replay step compares its headline numbers with the saved ones in `outputs/benchmark_1d/` and stops with an error if they differ by more than 1e-9.

Add `--with-live-2d` to also run the paper's 121 × 81 two-dimensional model. That step downloads the Census and BLS files (about 10 MB, cached afterwards) and adds about six minutes.

## Scripts

| Script | What it does |
|---|---|
| `scripts/before_factory_opens_main_model.py` | Downloads the data, calibrates the 1D model, and solves the forward-looking, surprise, and delayed-opening paths |
| `scripts/before_factory_opens_replay_1d_outputs.py` | Re-solves the 1D model from the saved 2021 inputs, with no download, and checks the result against the saved headline numbers |
| `scripts/before_factory_opens_robustness.py` | Sensitivity and numerical checks (Appendix C) |
| `scripts/before_factory_opens_2d_geographic_extension.py` | Two-dimensional version of the model (Appendix B) |
| `scripts/before_factory_opens_generate_figures.py` | Draws every figure from saved model output |
| `tests/test_replication.py` | Ten-second checks of the solver, the baseline calibration, and the saved headline numbers |
| `run_full_replication.sh` | Runs the tests, replay, robustness suite, and figures in order |

Every script has `--help` and runs with no arguments from the repository root. The 2D script is the exception: its default grid is a coarse 21 × 15 that solves in seconds, and the paper's results need `--nx 121 --ny 81 --dt 0.10`. `--demo` runs on synthetic data and is for testing code only.

## Where the paper's numbers come from

Everything the scripts write goes under `outputs/`. Only `outputs/benchmark_1d/` is tracked in git.

| In the paper | File |
|---|---|
| Headline 1D results (Table 3, Section 5.1) | `outputs/replayed_1d_results/Benchmark_1D_Headline_Metrics.json` |
| Sequential revisions (Section 5.3) | `outputs/replayed_1d_results/Benchmark_1D_Revision_Validation.json` |
| 1D numerical diagnostics (Table C2) | `outputs/replayed_1d_results/Benchmark_1D_Validation.json` |
| Robustness ranges (Table C1) | `outputs/robustness/Table_C1_Robustness_Summary.csv`, with one CSV per suite beside it |
| 2D results (Tables 3, B1, C2) | `outputs/preferred_2d_results/Preferred_2D_*.json` |
| Matched 1D comparison (Table 3) | `outputs/preferred_2d_results/Matched_1D_vs_Preferred_2D_Comparison.json` |
| Figures 1–3, B1, B2 | `outputs/figures/` |
| LODES shares for 2021–2023 (Table 1) | `outputs/robustness/10_Descriptive_LODES_Checks/`, written when the robustness script is run with `--holdout` |

Table B1 reports seven grids. Each row is one run of the 2D script with that row's `--nx`, `--ny`, and `--dt`; give each its own `--output-dir`.

`outputs/robustness/Robustness_Report.json` also holds a Richardson extrapolation of the headline ratio in the time step. The ratio is 44.5, 46.7, and 48.0 percent at time steps of 0.20, 0.10, and 0.05 years, and the extrapolation puts the limit near 49 to 50 percent. It is an estimate from three runs, not a solved specification.

## Reproducibility

The saved headline numbers were produced on one machine. A replay elsewhere agrees with them to about twelve decimal places; the difference is floating-point noise from the linear solver, and it moves diagnostics that are already near machine precision (the baseline fit, for example) by a similar absolute amount.

Last tested with Python 3.12, NumPy 2.4, SciPy 1.18, pandas 3.0, and matplotlib 3.11.

The robustness script has a `--stress` option for three harder cases (σ² = 0.10, γ = 0.5, and doubled κ). They are kept out of the reported ranges. A case that fails to converge is written to its CSV as a failure with the solver's error message.

## Data

All inputs are public.

- **Census LEHD LODES** (Ohio, 2021): home locations of workers along the Intel–Columbus corridor
- **BLS QCEW** (Licking County, 2021): county employment and average pay
- **Intel and JobsOhio announcements** (2022–2025): announced jobs, pay, and opening dates; sources are listed in `before_factory_opens_main_model.py`

`outputs/benchmark_1d/Benchmark_1D_Calibration.json` holds the 2021 inputs from the live run. The main model and the 2D script download fresh copies into `.hps_mfg_cache/` and reuse them on later runs; `--refresh` downloads them again.

## Citation

```bibtex
@techreport{stephens2026factory,
  author      = {Stephens, Harper},
  title       = {Before the Factory Opens: Anticipation and Spatial Adjustment to Announced and Delayed Local Shocks},
  institution = {University of Louisville},
  type        = {Working Paper},
  year        = {2026},
  note        = {Available at SSRN},
  doi         = {10.2139/ssrn.7500178},
  url         = {https://dx.doi.org/10.2139/ssrn.7500178}
}
```

## License

MIT. Contact: hpstep01@louisville.edu
