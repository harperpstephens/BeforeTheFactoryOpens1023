#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

python scripts/before_factory_opens_replay_1d_outputs.py \
  --preserved-output outputs/benchmark_1d \
  --output-dir outputs/replayed_1d_results

python scripts/before_factory_opens_robustness.py \
  --pipeline-output outputs/benchmark_1d \
  --output-dir reproduced_robustness_tests

python scripts/before_factory_opens_generate_figures.py \
  --one-d-output outputs/replayed_1d_results \
  --output-dir regenerated_figures

if [[ "${1:-}" == "--with-live-2d" ]]; then
  python scripts/before_factory_opens_2d_geographic_extension.py \
    --output-dir outputs/preferred_2d_results \
    --nx 121 --ny 81 --dt 0.10 \
    --one-d-headline outputs/benchmark_1d/Benchmark_1D_Headline_Metrics.json
  python scripts/before_factory_opens_generate_figures.py \
    --one-d-output outputs/replayed_1d_results \
    --two-d-output outputs/preferred_2d_results \
    --output-dir regenerated_figures
fi
