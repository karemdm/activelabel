#!/usr/bin/env sh
# Run activelabel with this folder's virtual environment (Linux, macOS). Example: ./activelabel.sh serve --open
here="$(cd "$(dirname "$0")" && pwd)"
exec "$here/.venv/bin/python" -m activelabel.cli "$@"
