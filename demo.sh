#!/usr/bin/env bash
# Simple helper to build demo store and launch SIGLA server with UI
set -e
STORE=demo_store

# Build store once if missing
if [ ! -f "$STORE.json" ]; then
    echo "[demo] создаётся индекс $STORE"
    python -m sigla ingest sample_capsules.json --model dummy -o "$STORE"
fi

# Start the server
python -m sigla serve -s "$STORE" -p 8000
