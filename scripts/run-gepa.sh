#!/usr/bin/env bash
# Run GEPA optimization on the server inside a zellij session.
# Execute on bombe: ./scripts/run-gepa.sh
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

SESSION="gepa"

# Create or attach to zellij session
if zellij list-sessions 2>/dev/null | grep -q "$SESSION"; then
  echo "Attaching to existing zellij session '$SESSION'..."
  zellij attach "$SESSION"
else
  echo "Starting GEPA optimization in zellij session '$SESSION'..."
  zellij --session "$SESSION"
  # Once inside zellij, run: ./scripts/run-gepa-inner.sh
fi
