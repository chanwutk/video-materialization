#!/usr/bin/env bash
# Run GEPA optimization on the server inside a zellij session.
# Execute on bombe: ./scripts/run-gepa.sh
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
cd ~/video-materialize

# Check API keys
if [[ -z "${GOOGLE_API_KEY:-}" ]] || [[ -z "${GEMINI_API_KEY:-}" ]]; then
  echo "ERROR: GOOGLE_API_KEY and GEMINI_API_KEY must be set."
  echo "Add them to ~/.bashrc or ~/.env:"
  echo '  export GOOGLE_API_KEY="your-key"'
  echo '  export GEMINI_API_KEY="your-key"'
  exit 1
fi

SESSION="gepa"

# Create or attach to zellij session
if zellij list-sessions 2>/dev/null | grep -q "$SESSION"; then
  echo "Attaching to existing zellij session '$SESSION'..."
  zellij attach "$SESSION"
else
  echo "Starting GEPA optimization in zellij session '$SESSION'..."
  zellij --session "$SESSION" -- bash -c '
    export PATH="$HOME/.local/bin:$PATH"
    cd ~/video-materialize

    echo "=== GEPA Optimization ==="
    echo "Model: gemini-2.5-flash (queries), gemini-2.5-flash-lite (routing)"
    echo "Max metric calls: 500"
    echo "Start time: $(date)"
    echo ""

    uv run python -m vm.main --gepa --gepa-max-calls 500 --model gemini-2.5-flash --concurrency 8

    echo ""
    echo "=== Optimization complete at $(date) ==="
    echo "Results: results/gepa/optimization_result.json"
    echo ""
    echo "To evaluate on test set, run:"
    echo "  uv run python -m vm.main --gepa-eval results/gepa/optimization_result.json --concurrency 8"
    echo ""
    echo "Press Enter to exit."
    read
  '
fi
