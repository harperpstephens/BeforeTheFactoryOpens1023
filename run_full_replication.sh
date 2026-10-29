#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
# Some systems (e.g. stock macOS) only have python3; set PYTHON to override.
PY="${PYTHON:-$(command -v python3 || command -v python)}"

"$PY" scripts/before_factory_opens_replay_1d_outputs.py \
  --preserved-output outputs/benchmark_1d \
  --output-dir outputs/replayed_1d_results

"$PY" scripts/before_factory_opens_robustness.py \
  --pipeline-output outputs/benchmark_1d \
  --output-dir reproduced_robustness_tests

"$PY" scripts/before_factory_opens_generate_figures.py \
  --one-d-output outputs/replayed_1d_results \
  --output-dir regenerated_figures

if [[ "${1:-}" == "--with-live-2d" ]]; then
  "$PY" scripts/before_factory_opens_2d_geographic_extension.py \
    --output-dir outputs/preferred_2d_results \
    --nx 121 --ny 81 --dt 0.10 \
    --one-d-headline outputs/benchmark_1d/Benchmark_1D_Headline_Metrics.json
  "$PY" scripts/before_factory_opens_generate_figures.py \
    --one-d-output outputs/replayed_1d_results \
    --two-d-output outputs/preferred_2d_results \
    --output-dir regenerated_figures
fi
