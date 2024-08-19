#!/bin/bash

# Check if inotifywait is installed
if ! command -v inotifywait &> /dev/null; then
  echo "inotifywait not found. Installing inotify-tools..."
  apt install -y inotify-tools 
fi

echo "Running test"
pytest -v "$@"

while inotifywait -e modify,create . ; do
  echo "Running test"
  pytest -v "$@" 
done
