#!/usr/bin/env bash

# Enable "strict mode" for bash.
# -e: Exit immediately if any command fails.
# -u: Exit immediately if you try to use an undeclared variable.
# -o pipefail: If a command within a pipeline (like A | B | C) fails, fail the whole pipeline, not just the last command.
set -euo pipefail

# Change the current working directory.
# $0 is the path to this script.
# dirname "$0" gets the folder this script is inside.
# /.. goes one level up.
# This ensures that no matter where you run this script from, it always executes relative to the project root.
cd "$(dirname "$0")/.."

# Initialize an empty bash array to hold the command we will eventually run.
compose_cmd=()

# Check if the newer Docker Compose V2 plugin is installed.
# We redirect standard output and standard error to /dev/null so the user doesn't see the version text.
if docker compose version >/dev/null 2>&1; then
  # If it succeeds, store the two-word command in our array.
  compose_cmd=(docker compose)
  
# If V2 isn't there, check if the older standalone docker-compose V1 is installed.
elif command -v docker-compose >/dev/null 2>&1; then
  # If it succeeds, store the single-word command in our array.
  compose_cmd=(docker-compose)
  
# If neither is found, yell at the user and exit.
else
  echo "Docker Compose is missing."
  echo "Install one of:"
  # Note: pacman is the package manager for Arch Linux (and derivatives like Manjaro).
  echo "  sudo pacman -S docker-compose"
  echo "or Docker Compose plugin for your Docker installation."
  # Exit with a status code of 1, indicating an error occurred.
  exit 1
fi

# Check if the Docker daemon (the background service) is running and accessible.
# 'docker info' will fail if the service is off or the user lacks permissions.
if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is not running or current user cannot access it."
  echo "Run:"
  # systemctl enable --now starts the service immediately and sets it to start on boot.
  echo "  sudo systemctl enable --now docker"
  # usermod adds the current user ($USER) to the 'docker' group to grant permissions.
  echo "  sudo usermod -aG docker $USER"
  echo "Then log out and log back in."
  exit 1
fi

# Print a friendly message showing exactly what command is about to be executed.
# ${compose_cmd[*]} prints the contents of the array as a single string.
echo "Starting PolySense with ${compose_cmd[*]} up --build"

# Execute the final command.
# "${compose_cmd[@]}" expands the array exactly as it was stored, preserving spaces.
# 'up' starts the containers, and '--build' forces Docker to rebuild the images first.
"${compose_cmd[@]}" up --build