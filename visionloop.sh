#!/usr/bin/env sh
# Run visionloop with this folder's virtual environment (Linux, macOS). Example: ./visionloop.sh serve --open
here="$(cd "$(dirname "$0")" && pwd)"
exec "$here/.venv/bin/python" -m visionloop.cli "$@"
