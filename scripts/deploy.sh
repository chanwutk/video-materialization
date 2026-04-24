#!/usr/bin/env bash
# Deploy code to bombe and set up the environment.
# Run from local laptop: ./scripts/deploy.sh
set -euo pipefail

SERVER="bombe"
REMOTE_DIR="/work/cwkt/projects/materialized-video"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Syncing code to $SERVER:$REMOTE_DIR ==="
rsync -avz --delete \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.git/' \
  --exclude '.cursor/' \
  --exclude '.claude/' \
  --exclude 'data/' \
  --exclude 'results/' \
  --exclude 'paper' \
  "$LOCAL_DIR/" "$SERVER:$REMOTE_DIR/"

echo ""
echo "=== Setting up server environment ==="
ssh "$SERVER" REMOTE_DIR="$REMOTE_DIR" bash -s <<'REMOTE'
set -euo pipefail
cd "$REMOTE_DIR"

# Install uv if not present
if ! command -v uv &>/dev/null; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# Install zellij if not present
if ! command -v zellij &>/dev/null; then
  echo "Installing zellij..."
  mkdir -p "$HOME/.local/bin"
  curl -LsSf https://github.com/zellij-org/zellij/releases/latest/download/zellij-x86_64-unknown-linux-musl.tar.gz \
    | tar xz -C "$HOME/.local/bin"
  chmod +x "$HOME/.local/bin/zellij"
fi

export PATH="$HOME/.local/bin:$PATH"

# Sync dependencies
echo "Installing Python dependencies..."
uv sync

echo ""
echo "Done. Environment ready at $REMOTE_DIR"
echo "uv: $(uv --version)"
echo "zellij: $(zellij --version)"
echo "python: $(uv run python --version)"
REMOTE

echo ""
echo "=== Deploy complete ==="
echo "Next: ssh $SERVER and run ./scripts/run-gepa.sh"
