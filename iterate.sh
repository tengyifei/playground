#!/bin/bash

set -e

MONITOR_DIR="/$(realpath -m --relative-to / ../pytorch/xla)"

echo "Monitoring $MONITOR_DIR"

pushd ../pytorch/xla/test

# Check if inotifywait is installed
if ! command -v inotifywait &> /dev/null; then
  echo "inotifywait not found. Installing inotify-tools..."
  apt install -y inotify-tools
fi

echo "Running test"
for arg in "$@"; do
  echo "Testing $arg"
  python3 "$arg" ||:
done
echo "=========== DONE ==========="

while inotifywait -r -e modify,create "$MONITOR_DIR" ; do
  echo "Running test"
  for arg in "$@"; do
    echo "Testing $arg"
    python3 "$arg" ||:
  done
  echo "=========== DONE ==========="
done

popd
