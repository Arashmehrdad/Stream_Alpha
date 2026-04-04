import argparse
import asyncio
import json
import re
from pathlib import Path

import asyncpg
import pyarrow as pa
import pyarrow.parquet as pq

from app.common.config import Settings

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def quote_identifier(identifier: str) -> str:
    if not _IDENTIFIER_RE.match(identifier):
        raise ValueError(f"Unsafe SQL identifier: {identifier}")
    return f'"{identifier}"'


def quote_table_name(name: str) -> str:
    parts = name.split(".")
    if not 1 <= len(parts) <= 2:
        raise ValueError(f"Unsupported table name format: {name}")
    return ".".join(quote_identifier(part) for part in parts)


def write_parquet_part(rows: list[dict], output_path: Path) -> int:
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output_path, compression="snappy")
    return table.num_rows


async def export_feature_ohlc(
    *,
    config_path: Path,
    output_root: Path,
    rows_per_file: int,
) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    source_table = str(config["source_table"])
    symbols = [str(symbol) for symbol in config["symbols"]]
    time_column = str(config.get("time_column", "as_of_time"))
    interval_column = str(config.get("interval_column", "interval_begin"))

    settings = Settings.from_env()
    dsn = settings.postgres.dsn

    output_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "config_path": str(config_path.resolve()),
        "source_table": source_table,
        "symbols": {},
        "rows_per_file": rows_per_file,
    }

    query = f"""
        SELECT *
        FROM {quote_table_name(source_table)}
        WHERE symbol = $1
        ORDER BY {quote_identifier(time_column)} ASC, {quote_identifier(interval_column)} ASC
    """

    connection = await asyncpg.connect(dsn)
    try:
        for symbol in symbols:
            safe_symbol = symbol.replace("/", "_")
            symbol_dir = output_root / safe_symbol
            symbol_dir.mkdir(parents=True, exist_ok=True)

            print(f"Exporting {symbol} -> {symbol_dir}")
            batch: list[dict] = []
            part_index = 0
            total_rows = 0

            async with connection.transaction():
                cursor = connection.cursor(query, symbol, prefetch=5000)
                async for record in cursor:
                    batch.append(dict(record))
                    if len(batch) >= rows_per_file:
                        part_path = symbol_dir / f"part-{part_index:05d}.parquet"
                        written = write_parquet_part(batch, part_path)
                        total_rows += written
                        print(f"  wrote {part_path.name} ({written} rows)")
                        batch.clear()
                        part_index += 1

            if batch:
                part_path = symbol_dir / f"part-{part_index:05d}.parquet"
                written = write_parquet_part(batch, part_path)
                total_rows += written
                print(f"  wrote {part_path.name} ({written} rows)")
                batch.clear()

            manifest["symbols"][symbol] = {
                "folder": str(symbol_dir.resolve()),
                "rows": total_rows,
                "parts": part_index + (1 if total_rows > 0 else 0),
            }

        manifest_path = output_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"\nDone. Manifest: {manifest_path.resolve()}")
    finally:
        await connection.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/training.m20.json",
        help="Path to the training config",
    )
    parser.add_argument(
        "--out",
        default="exports/feature_ohlc_for_colab",
        help="Output folder for parquet shards",
    )
    parser.add_argument(
        "--rows-per-file",
        type=int,
        default=200000,
        help="Number of rows per parquet file",
    )
    args = parser.parse_args()

    asyncio.run(
        export_feature_ohlc(
            config_path=Path(args.config),
            output_root=Path(args.out),
            rows_per_file=int(args.rows_per_file),
        )
    )


if __name__ == "__main__":
    main()
