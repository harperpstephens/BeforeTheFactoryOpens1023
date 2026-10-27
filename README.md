# Before the Factory Opens

Replication code for "[Before the Factory Opens: Anticipation and Spatial Adjustment to Announced and Delayed Local Shocks](https://dx.doi.org/10.2139/ssrn.7500178)" by Harper Stephens (working paper, 2026).

The paper uses a spatial mean field game, calibrated to 2021 Census LODES residences, to measure how much workers relocate in anticipation of Intel's Ohio semiconductor plant before it opens.

## Scripts

| Script | What it does |
|---|---|
| `scripts/before_factory_opens_main_model.py` | Downloads the data, calibrates the 1D model, solves the forward-looking, surprise, and delayed-opening paths, and writes the headline results and Figure 2 |
| `scripts/before_factory_opens_replay_1d_outputs.py` | Re-solves the 1D model from the saved 2021 inputs in `outputs/benchmark_1d/`, with no download |
| `scripts/before_factory_opens_robustness.py` | Sensitivity and numerical checks (Appendix C) |
| `scripts/before_factory_opens_2d_geographic_extension.py` | Two-dimensional version of the model (Appendix B) |
| `scripts/before_factory_opens_generate_figures.py` | Draws Figures 1–3 and B1a/B2 from model output |
| `run_full_replication.sh` | Runs the replay, robustness checks, and figures in order (`--with-live-2d` adds the 2D model) |

Every script has `--help`.

## Usage

```bash
pip install -r requirements.txt

python scripts/before_factory_opens_replay_1d_outputs.py \
  --preserved-output outputs/benchmark_1d \
  --output-dir outputs/replayed_1d_results
```

The results should match `outputs/benchmark_1d/Benchmark_1D_Headline_Metrics.json`.

The paper's 2D results use `--nx 121 --ny 81 --dt 0.10`. The `--demo` option runs on synthetic data and is for testing code only.

## Data

All inputs are public.

- **Census LEHD LODES** (Ohio, 2021): home locations of workers along the Intel–Columbus corridor
- **BLS QCEW** (Licking County, 2021): county employment and average pay
- **Intel and JobsOhio announcements** (2022–2025): announced jobs, pay, and opening dates; sources are listed in `before_factory_opens_main_model.py`

`outputs/benchmark_1d/Benchmark_1D_Calibration.json` holds the 2021 inputs from the live run. The main model downloads fresh copies.

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
