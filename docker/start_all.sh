#!/usr/bin/env bash
set -euo pipefail

cd /workspace

cleanup() {
  if [[ -n "${BACKEND_PID:-}" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    kill "$BACKEND_PID" || true
  fi
  if [[ -n "${FRONTEND_PID:-}" ]] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
    kill "$FRONTEND_PID" || true
  fi
}

trap cleanup EXIT INT TERM

echo "[PolySense] Starting backend on 0.0.0.0:8000"
python -m uvicorn app.backend.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

echo "[PolySense] Starting frontend on 0.0.0.0:5173"
python -m http.server 5173 --bind 0.0.0.0 --directory app/frontend &
FRONTEND_PID=$!

wait -n "$BACKEND_PID" "$FRONTEND_PID"
exit $?
