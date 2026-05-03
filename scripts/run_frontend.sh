#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../app/frontend"
python3 -m http.server 5173 --bind 127.0.0.1
