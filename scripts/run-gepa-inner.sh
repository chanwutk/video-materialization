#!/usr/bin/env bash
# Run GEPA optimization. Execute this inside a zellij/tmux session on bombe.
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
cd /work/cwkt/projects/materialized-video

# Check API keys
if [[ -z "${GOOGLE_API_KEY:-}" ]] || [[ -z "${GEMINI_API_KEY:-}" ]]; then
  echo "ERROR: GOOGLE_API_KEY and GEMINI_API_KEY must be set."
  echo "Add them to ~/.bashrc or ~/.env:"
  echo '  export GOOGLE_API_KEY="your-key"'
  echo '  export GEMINI_API_KEY="your-key"'
  exit 1
fi

echo "=== GEPA Optimization ==="
echo "Model: gemini-2.5-flash (queries), gemini-2.5-flash-lite (routing)"
echo "Max metric calls: 500"
echo "Start time: $(date)"
echo ""

# Outer retry loop: if the process dies (e.g., from exhausting in-process retries
# during a sustained API outage), restart up to MAX_RESTARTS times. Disk caches
# (routing, answers, GEPA evaluation cache) preserve previous progress.
MAX_RESTARTS=5
attempt=0
while (( attempt < MAX_RESTARTS )); do
  attempt=$(( attempt + 1 ))
  echo ""
  echo "--- Run attempt $attempt/$MAX_RESTARTS at $(date) ---"
  if uv run python -m vm.main --gepa --gepa-max-calls 500 --model gemini-2.5-flash --concurrency 8; then
    echo ""
    echo "=== Optimization complete at $(date) ==="
    break
  else
    rc=$?
    echo ""
    echo "!!! Run failed with exit code $rc at $(date) !!!"
    if (( attempt < MAX_RESTARTS )); then
      sleep_s=60
      echo "Sleeping ${sleep_s}s then restarting..."
      sleep "$sleep_s"
    else
      echo "Exhausted $MAX_RESTARTS attempts; giving up."
      exit "$rc"
    fi
  fi
done

echo ""
echo "Results: results/gepa/optimization_result.json"
echo ""
echo "To evaluate on test set, run:"
echo "  uv run python -m vm.main --gepa-eval results/gepa/optimization_result.json --concurrency 8"
