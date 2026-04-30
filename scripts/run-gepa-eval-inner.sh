#!/usr/bin/env bash
# Evaluate the GEPA-learned directive on the test split alongside baselines.
# Execute inside zellij/tmux on bombe: ./scripts/run-gepa-eval-inner.sh
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
cd /work/cwkt/projects/materialized-video

if [[ -z "${GOOGLE_API_KEY:-}" ]] || [[ -z "${GEMINI_API_KEY:-}" ]]; then
  echo "ERROR: GOOGLE_API_KEY and GEMINI_API_KEY must be set."
  exit 1
fi

echo "=== GEPA Test-Set Evaluation ==="
echo "Start time: $(date)"
echo ""

# Outer retry loop, same as run-gepa-inner.sh.
MAX_RESTARTS=5
attempt=0
while (( attempt < MAX_RESTARTS )); do
  attempt=$(( attempt + 1 ))
  echo "--- Run attempt $attempt/$MAX_RESTARTS at $(date) ---"
  if uv run python -m vm.main \
       --gepa-eval results/gepa/optimization_result.json \
       --model gemini-2.5-flash \
       --concurrency 8; then
    echo ""
    echo "=== Evaluation complete at $(date) ==="
    break
  else
    rc=$?
    echo "!!! Run failed with exit code $rc at $(date) !!!"
    if (( attempt < MAX_RESTARTS )); then
      echo "Sleeping 60s then restarting..."
      sleep 60
    else
      echo "Exhausted $MAX_RESTARTS attempts; giving up."
      exit "$rc"
    fi
  fi
done

echo ""
echo "Results: results/gepa_eval/"
echo "  - token_breakdown.csv"
echo "  - raw_results.json"
echo "  - tokens_vs_accuracy.png"
echo "  - tokens_vs_accuracy_query.png"
