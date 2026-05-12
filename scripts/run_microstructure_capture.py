"""Dry-run isolated research microstructure capture service helper."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.training.microstructure_capture_service import (  # pylint: disable=wrong-import-position
    DEFAULT_OUTPUT_DIR,
    write_microstructure_capture_service_dry_run,
)


def main() -> None:
    """Write the dry-run capture service artifact."""
    parser = argparse.ArgumentParser(description="Dry-run microstructure capture service")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--symbols", default="BTC/USD")
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--duration-seconds", type=int, default=60)
    parser.add_argument("--max-events", type=int, default=1000)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dsn", default="")
    parser.add_argument("--ws-url", default="wss://ws.kraken.com/v2")
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    symbols = tuple(part.strip() for part in args.symbols.split(",") if part.strip())
    result = write_microstructure_capture_service_dry_run(
        repo_root=repo_root,
        output_dir=output_dir,
        symbols=symbols,
        depth=args.depth,
        duration_seconds=args.duration_seconds,
        max_events=args.max_events,
        execute=args.execute,
        dsn=args.dsn or None,
        ws_url=args.ws_url,
    )
    print(f"output_dir={Path(result['output_files']['manifest_json']).parent}")
    print(f"capture_service_status={result['capture_service_status']}")
    print(f"network_capture_executed={result['network_capture_executed']}")
    print(f"recommendation={result['recommendation']}")
    print(f"next_required_action={result['next_required_action']}")


if __name__ == "__main__":
    main()
