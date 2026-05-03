#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

compose_cmd=()
if docker compose version >/dev/null 2>&1; then
  compose_cmd=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  compose_cmd=(docker-compose)
else
  echo "Docker Compose is missing."
  echo "Install one of:"
  echo "  sudo pacman -S docker-compose"
  echo "or Docker Compose plugin for your Docker installation."
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is not running or current user cannot access it."
  echo "Run:"
  echo "  sudo systemctl enable --now docker"
  echo "  sudo usermod -aG docker $USER"
  echo "Then log out and log back in."
  exit 1
fi

echo "Starting PolySense with ${compose_cmd[*]} up --build"
"${compose_cmd[@]}" up --build
