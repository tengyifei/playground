#!/bin/sh

while inotifywait -e modify,create . ; do
  echo "Running test"
  pytest -v
done
