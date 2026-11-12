"""Before the Factory Opens: draw the paper's figures from saved model outputs.

No model is solved here; the script only reads results that are already on disk.

- Figures 1-3 need a 1D output folder, from before_factory_opens_main_model.py
  or the offline replay script.
- Appendix Figures B1a, B1b, and B2 need a 2D output folder, from
  before_factory_opens_2d_geographic_extension.py.

With no arguments the script looks in outputs/replayed_1d_results and
outputs/preferred_2d_results, draws whatever it finds, and saves the PNGs to
outputs/figures.

Usage (from the repository root):
    python3 scripts/before_factory_opens_generate_figures.py
    python3 scripts/before_factory_opens_generate_figures.py --one-d-output outputs/live_1d_results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files; no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"
SCENARIO_LABELS = {"forward_looking": "Forward-looking", "surprise_opening": "Surprise at opening"}
DPI = 220


def line_axes(figsize):
    """A figure for line plots, without the top and right frame lines."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax


def mark_time(ax, t, label, ls="--"):
    """A vertical line at time ``t`` with a small label running up from the axis."""
    ax.axvline(t, ls=ls, lw=1.2, alpha=0.75)
    ax.text(t, 0.0, f" {label}", rotation=90, va="bottom", ha="left", fontsize=8,
            transform=ax.get_xaxis_transform())


def save(fig, dest: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(dest / name, dpi=DPI)
    plt.close(fig)
    print(f"wrote {dest / name}")


def one_d(out: Path, dest: Path) -> None:
    """Figures 1-3: baseline residences, anticipation, and the sequential revisions."""
    cal = json.loads((out / "Benchmark_1D_Calibration.json").read_text(encoding="utf-8"))
    revision = json.loads((out / "Benchmark_1D_Revision_Validation.json").read_text(encoding="utf-8"))
    base = pd.read_csv(out / "Benchmark_1D_Baseline.csv")
    paths = pd.read_csv(out / "Benchmark_1D_Paths.csv")
    seq = pd.read_csv(out / "Benchmark_1D_Sequential_Revision_Path.csv").sort_values("time")
    raw = np.asarray(cal["data"]["raw_mass"], float)
    opening = float(cal["model"]["opening"])
    x_miles = base["x"].to_numpy(float) * float(cal["calibration"]["half_length_miles"])
    forward = paths[paths.scenario == "forward_looking"].sort_values("time")
    surprise = paths[paths.scenario == "surprise_opening"].sort_values("time")
    band_label = "Residence mass in projected ±5-mile band (%)"

    # Figure 1: raw and smoothed 2021 residence shares along the corridor.
    fig, ax = line_axes((8.8, 4.55))
    ax.plot(x_miles, 100 * raw, lw=1.2, label="Raw 2021 projected residence mass")
    ax.plot(x_miles, 100 * base["smoothed_target_mass"], lw=2.4, label="Smoothed calibration target")
    ax.axvline(0, ls="--", lw=1, alpha=0.75)
    ax.set_xlim(x_miles.min(), x_miles.max())
    ax.set_xlabel("Miles along Intel-Columbus axis")
    ax.set_ylabel("Cell share of retained residences (%)")
    ax.legend(frameon=False, loc="upper left")
    save(fig, dest, "Figure_1_Baseline_Residence_Mass_1D.png")

    # Figure 2: mass near the site over time, forward-looking vs. surprise.
    fig, ax = line_axes((8.8, 4.8))
    ax.plot(forward.time, 100 * forward.band_mass, lw=2.4, label=SCENARIO_LABELS["forward_looking"])
    ax.plot(surprise.time, 100 * surprise.band_mass, lw=2.4, label=SCENARIO_LABELS["surprise_opening"])
    mark_time(ax, opening, "Initial expected opening")
    ax.set_xlim(0, forward.time.max())
    ax.set_xlabel("Years since January 2022")
    ax.set_ylabel(band_label)
    ax.legend(frameon=False, loc="lower right")
    save(fig, dest, "Figure_2_Anticipatory_Adjustment_1D.png")

    # Figure 3: the same paths with the 2024 and 2025 delays layered on. The
    # dotted lines sit at the grid times where the model applies each revision.
    fig, ax = line_axes((8.8, 4.8))
    ax.plot(forward.time, 100 * forward.band_mass, lw=2.4, label="Original 2022 information")
    ax.plot(seq.time, 100 * seq.band_mass, lw=2.4, label="Sequential revisions")
    ax.plot(surprise.time, 100 * surprise.band_mass, lw=2.0, ls="--", label="Surprise at opening")
    for event in revision["events"]:
        mark_time(ax, float(event["news_grid"]), f"{event['label'][:4]} news", ls=":")
    mark_time(ax, opening, "Initial opening horizon")
    ax.set_xlim(0, forward.time.max())
    ax.set_xlabel("Years since January 2022")
    ax.set_ylabel(band_label)
    ax.legend(frameon=False, loc="lower right")
    save(fig, dest, "Figure_3_Sequential_Public_Revisions_1D.png")


def corridor_map(arr, cal, label, **kw):
    """One cell-level map of the corridor, with the Intel site marked."""
    half, cross = float(cal["half_length_miles"]), float(cal["cross_halfwidth_miles"])
    fig, ax = plt.subplots(figsize=(8, 5.6))
    im = ax.imshow(arr, origin="lower", aspect="auto", extent=[-half, half, -cross, cross], **kw)
    fig.colorbar(im, ax=ax, label=label)
    ax.scatter([0], [0], marker="x", s=60, label="Intel site")
    ax.set_xlabel("Miles along Intel-Columbus axis")
    ax.set_ylabel("Miles perpendicular to axis")
    return fig, ax


def two_d(out: Path, dest: Path) -> None:
    """Appendix Figures B1a (baseline map), B1b (opening difference map), and B2 (paths)."""
    cal = json.loads((out / "Preferred_2D_Calibration.json").read_text(encoding="utf-8"))
    base = pd.read_csv(out / "Preferred_2D_Baseline.csv")
    paths = pd.read_csv(out / "Preferred_2D_Paths.csv")
    c, m = cal["calibration"], cal["model"]
    shape = (int(m["ny"]), int(m["nx"]))

    # Figure B1a: smoothed 2021 baseline on the 2D grid.
    fig, ax = corridor_map(base["smoothed_target_mass"].to_numpy(float).reshape(shape), c, "Cell probability")
    ax.legend()
    save(fig, dest, "Figure_B1a_Smoothed_2021_Baseline_2D.png")

    # Figure B1b: where the forward-looking and surprise populations differ at opening.
    if "opening_gap_mass" in base:
        diff = base["opening_gap_mass"].to_numpy(float).reshape(shape)
        vmax = float(np.max(np.abs(diff)))
        fig, ax = corridor_map(diff, c, "Forward-looking minus surprise cell mass",
                               cmap="coolwarm", vmin=-vmax, vmax=vmax)
        save(fig, dest, "Figure_B1b_Opening_Distribution_Difference_2D.png")
    else:
        print("skipped Figure B1b: this 2D output predates the opening_gap_mass column; rerun the 2D script to draw it")

    # Figure B2: mass inside the true five-mile circle over time.
    fig, ax = line_axes((8, 5))
    for name in SCENARIO_LABELS:
        d = paths[paths.scenario == name].sort_values("time")
        ax.plot(d.time, d.circle_mass, lw=2.0, label=SCENARIO_LABELS[name])
    ax.axvline(float(m["opening"]), ls="--", lw=1.2, alpha=0.75)
    ax.set_xlabel("Years since January 2022")
    ax.set_ylabel("Mass within true 5-mile circle")
    ax.legend(frameon=False, loc="lower right")
    save(fig, dest, "Figure_B2_Anticipatory_Adjustment_2D.png")


def main() -> None:
    ap = argparse.ArgumentParser(description="Draw the paper's figures from saved model outputs.")
    ap.add_argument("--one-d-output", type=Path,
                    help="Output folder from the 1D model or replay (default: outputs/replayed_1d_results, if present).")
    ap.add_argument("--two-d-output", type=Path,
                    help="Output folder from the 2D extension (default: outputs/preferred_2d_results, if present).")
    ap.add_argument("--output-dir", type=Path, default=OUTPUTS / "figures",
                    help="Where to save the PNGs (default: outputs/figures).")
    a = ap.parse_args()

    one, two = a.one_d_output, a.two_d_output
    if one is None and two is None:
        one = next((d for d in (OUTPUTS / "replayed_1d_results",) if d.is_dir()), None)
        two = next((d for d in (OUTPUTS / "preferred_2d_results",) if d.is_dir()), None)
        if one is None and two is None:
            ap.error("no saved outputs found; run the replay script first, or pass --one-d-output / --two-d-output")

    a.output_dir.mkdir(parents=True, exist_ok=True)
    if one:
        one_d(one, a.output_dir)
    if two:
        two_d(two, a.output_dir)


if __name__ == "__main__":
    main()
