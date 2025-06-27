#!/usr/bin/env bash
# Serve the glass shader demo via a simple HTTP server
set -e
cd "$(dirname "$0")"
python -m http.server 8080
