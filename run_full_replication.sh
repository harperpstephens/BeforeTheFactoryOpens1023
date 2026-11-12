#!/usr/bin/env bash
# Reproduce the paper's 1D results, robustness tables, and figures from the
# saved 2021 inputs. Nothing is downloaded unless --with-live-2d is given.
#
#   ./run_full_replication.sh                 1D replay, robustness suite, Figures 1-3  (about 2 minutes)
#   ./run_full_replication.sh --with-live-2d  also the paper's 121 x 81 2D run and Figures B1-B2  (about 6 more minutes)
#
# Everything is written under outputs/.
set -euo pipefail

cd "$(dirname "$0")"
# Some systems (e.g. stock macOS) only have python3; set PYTHON to override.
PY="${PYTHON:-$(command -v python3 || command -v python)}"

"$PY" tests/test_replication.py
"$PY" scripts/before_factory_opens_replay_1d_outputs.py
"$PY" scripts/before_factory_opens_robustness.py

if [[ "${1:-}" == "--with-live-2d" ]]; then
  "$PY" scripts/before_factory_opens_2d_geographic_extension.py --nx 121 --ny 81 --dt 0.10 \
    --one-d-headline outputs/benchmark_1d/Benchmark_1D_Headline_Metrics.json
fi

"$PY" scripts/before_factory_opens_generate_figures.py
