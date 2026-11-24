"""Command line: ``activelabel serve``."""

from __future__ import annotations

import argparse
import sys
import webbrowser
from collections.abc import Sequence

import uvicorn

from activelabel.web.app import create_app


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: the arguments after the program name; None reads ``sys.argv``.

    Returns:
        The exit code, 0 when the server stopped normally.
    """
    parser = argparse.ArgumentParser(prog="activelabel", description="Image datasets from curation to models")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="Run the web interface")
    serve.add_argument(
        "--host", default="127.0.0.1", help="Use 0.0.0.0 only on a trusted network (there is no login)."
    )
    serve.add_argument("--port", type=int, default=8090)
    serve.add_argument("--open", action="store_true", help="Open the browser")
    args = parser.parse_args(argv)
    if args.command == "serve":
        url = f"http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}"
        print(f"activelabel: {url}", file=sys.stderr)
        if args.open:
            webbrowser.open(url)
        uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
