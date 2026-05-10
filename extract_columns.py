#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def err(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a subset of columns from a CSV file."
    )
    parser.add_argument("input_csv", help="Input CSV path.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument(
        "--columns",
        required=True,
        nargs="+",
        metavar="COLUMN",
        help="One or more column names to extract.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_csv).resolve()
    output_path = Path(args.output).resolve()

    if not input_path.exists():
        err(f"Input file does not exist: {input_path}")
        return 1

    with open(input_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            err("CSV has no header.")
            return 1
        fieldnames = list(reader.fieldnames)

        missing = [c for c in args.columns if c not in fieldnames]
        if missing:
            err(f"Columns not found in CSV: {', '.join(missing)}")
            err(f"Available columns: {', '.join(fieldnames)}")
            return 1

        rows = list(reader)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=args.columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows with columns [{', '.join(args.columns)}] to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
