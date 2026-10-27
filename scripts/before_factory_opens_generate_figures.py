"""Before the Factory Opens: redraw the paper's figures from saved model outputs.

No model is solved here; the script only reads results that are already on disk.

- 1D figures (Figures 1-3) need an output folder from
  before_factory_opens_main_model.py or the offline replay script.
- 2D figures (Appendix Figures B1a and B2) need an output folder from
  before_factory_opens_2d_geographic_extension.py.

Each file is saved with a "_Regenerated" suffix, so it never overwrites the
original figure assets.

Usage:
    python before_factory_opens_generate_figures.py --one-d-output outputs/replayed_1d_results \\
        --output-dir regenerated_figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def one_d(out: Path, dest: Path) -> None:
    """Figures 1-3: baseline residences, anticipation, and the sequential revisions."""
    cal = json.loads((out / "Benchmark_1D_Calibration.json").read_text(encoding="utf-8"))
    raw = np.asarray(cal["data"]["raw_mass"], float)
    base = pd.read_csv(out / "Benchmark_1D_Baseline.csv")
    paths = pd.read_csv(out / "Benchmark_1D_Paths.csv")
    seq = pd.read_csv(out / "Benchmark_1D_Sequential_Revision_Path.csv")
    model = cal["model"]
    half = float(cal["calibration"]["half_length_miles"])
    x_miles = base["x"].to_numpy(float) * half

    # Figure 1: raw and smoothed 2021 residence shares along the corridor.
    fig, ax = plt.subplots(figsize=(8.8, 4.55))
    ax.plot(x_miles, 100 * raw, label="Raw 2021 projected residence mass")
    ax.plot(x_miles, 100 * base["smoothed_target_mass"], label="Smoothed calibration target")
    ax.axvline(0, ls="--", lw=1)
    ax.set_xlabel("Miles along Intel-Columbus axis")
    ax.set_ylabel("Cell share of retained residences (%)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(dest / "Figure_1_Baseline_Residence_Mass_1D_Regenerated.png", dpi=220)
    plt.close(fig)

    # Figure 2: mass near the site over time, forward-looking vs. surprise.
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    labels = {"forward_looking": "Forward-looking", "surprise_opening": "Surprise at opening"}
    for name, d in paths.groupby("scenario"):
        ax.plot(d["time"], 100 * d["band_mass"], label=labels.get(name, name))
    ax.axvline(float(model["opening"]), ls="--", lw=1)
    ax.set_xlabel("Years since January 2022")
    ax.set_ylabel("Residence mass in projected +/-5-mile band (%)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(dest / "Figure_2_Anticipatory_Adjustment_1D_Regenerated.png", dpi=220)
    plt.close(fig)

    # Figure 3: the same paths with the 2024 and 2025 delays layered on.
    forward = paths[paths.scenario == "forward_looking"].sort_values("time")
    surprise = paths[paths.scenario == "surprise_opening"].sort_values("time")
    seq = seq.sort_values("time")
    c = cal["calibration"]
    from datetime import date

    def years(a: str, b: str) -> float:
        """Years between two ISO dates (same rule as the model script)."""
        return (date.fromisoformat(b) - date.fromisoformat(a)).days / 365.2425

    r1 = years(c["announcement"], c["revision1_date"])
    r2 = years(c["announcement"], c["revision2_date"])

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(forward.time, 100 * forward.band_mass, label="Original 2022 information")
    ax.plot(seq.time, 100 * seq.band_mass, label="Sequential revisions")
    ax.plot(surprise.time, 100 * surprise.band_mass, ls="--", label="Surprise at opening")
    ax.axvline(r1, ls=":", lw=1)
    ax.axvline(r2, ls=":", lw=1)
    ax.axvline(float(model["opening"]), ls="--", lw=1)
    ax.set_xlabel("Years since January 2022")
    ax.set_ylabel("Residence mass in projected +/-5-mile band (%)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(dest / "Figure_3_Sequential_Public_Revisions_1D_Regenerated.png", dpi=220)
    plt.close(fig)


def two_d(out: Path, dest: Path) -> None:
    """Appendix Figures B1a (2D baseline map) and B2 (2D anticipation paths)."""
    cal = json.loads((out / "Preferred_2D_Calibration.json").read_text(encoding="utf-8"))
    base = pd.read_csv(out / "Preferred_2D_Baseline.csv")
    paths = pd.read_csv(out / "Preferred_2D_Paths.csv")
    c = cal["calibration"]
    m = cal["model"]
    nx, ny = int(m["nx"]), int(m["ny"])
    half, cross = float(c["half_length_miles"]), float(c["cross_halfwidth_miles"])

    # Figure B1a: smoothed 2021 baseline on the 2D grid.
    arr = base["smoothed_target_mass"].to_numpy(float).reshape(ny, nx)
    fig, ax = plt.subplots(figsize=(8, 5.6))
    im = ax.imshow(arr, origin="lower", aspect="auto", extent=[-half, half, -cross, cross])
    fig.colorbar(im, ax=ax, label="Cell probability")
    ax.scatter([0], [0], marker="x", s=60, label="Intel site")
    ax.set_xlabel("Miles along Intel-Columbus axis")
    ax.set_ylabel("Miles perpendicular to axis")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(dest / "Figure_B1a_Smoothed_2021_Baseline_2D_Regenerated.png", dpi=220)
    plt.close(fig)

    opening = float(m["opening"])
    f = paths[(paths.scenario == "forward_looking") & np.isclose(paths.time, opening)]
    s = paths[(paths.scenario == "surprise_opening") & np.isclose(paths.time, opening)]
    # Figure B1b (the cell-by-cell difference at opening) can't be redrawn
    # here, because Preferred_2D_Paths.csv doesn't store cell-level mass. Use
    # the PNG the 2D solver writes. The timing paths below are fully reproducible.

    # Figure B2: mass inside the true five-mile circle over time.
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = {"forward_looking": "Forward-looking", "surprise_opening": "Surprise at opening"}
    for name, d in paths.groupby("scenario"):
        ax.plot(d.time, d.circle_mass, label=labels.get(name, name))
    ax.axvline(opening, ls="--", lw=1)
    ax.set_xlabel("Years since Jan. 2022")
    ax.set_ylabel("Mass within true 5-mile circle")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(dest / "Figure_B2_Anticipatory_Adjustment_2D_Regenerated.png", dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Redraw the paper's figures from saved model outputs.")
    ap.add_argument("--one-d-output", type=Path, help="Output folder from the 1D model or replay run.")
    ap.add_argument("--two-d-output", type=Path, help="Output folder from the 2D extension.")
    ap.add_argument("--output-dir", type=Path, required=True, help="Where to save the redrawn figures.")
    a = ap.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if a.one_d_output:
        one_d(a.one_d_output, a.output_dir)
    if a.two_d_output:
        two_d(a.two_d_output, a.output_dir)
    if not a.one_d_output and not a.two_d_output:
        ap.error("provide --one-d-output and/or --two-d-output")


if __name__ == "__main__":
    main()
