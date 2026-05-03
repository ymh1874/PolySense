#!/usr/bin/env bash

# Enable bash strict mode.
# -e: Exit immediately if any command returns a non-zero (failure) status.
# -u: Exit immediately if an uninitialized variable is used.
# -o pipefail: Ensure errors in pipelines (e.g., A | B) cause the whole line to fail.
set -euo pipefail

# Change the working directory to the parent folder of wherever this script is located.
# This ensures the script can find your 'docker-compose.yml' file regardless of where you run it from.
cd "$(dirname "$0")/.."

# Check if the newer Docker Compose V2 plugin is installed.
# Output is hidden by redirecting it to /dev/null.
if docker compose version >/dev/null 2>&1; then
  # If V2 is found, use it to stop and remove the containers, networks, and volumes defined in your compose file.
  docker compose down
  
# If V2 isn't found, check for the older standalone V1 (docker-compose).
elif command -v docker-compose >/dev/null 2>&1; then
  # If V1 is found, execute its teardown command.
  docker-compose down
  
# If neither version of Docker Compose is installed on the system...
else
  # Print an error message to the terminal.
  echo "Docker Compose is missing."
  # Exit the script with a status code of 1, signaling to the OS that it failed.
  exit 1
fi