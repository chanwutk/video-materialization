#!/usr/bin/env bash
# Pull results and cache back from bombe to local laptop.
# Run from local laptop: ./scripts/sync-results.sh
set -euo pipefail

SERVER="bombe"
REMOTE_DIR="~/video-materialize"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Syncing results from $SERVER ==="
rsync -avz "$SERVER:$REMOTE_DIR/results/" "$LOCAL_DIR/results/"

echo ""
echo "=== Syncing cache from $SERVER ==="
rsync -avz "$SERVER:$REMOTE_DIR/data/cache/" "$LOCAL_DIR/data/cache/"

echo ""
echo "=== Done ==="
echo "Results: $LOCAL_DIR/results/"
echo "Cache:   $LOCAL_DIR/data/cache/"
