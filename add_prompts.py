#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import types

DEFAULT_LIMIT = 0
DEFAULT_DELAY_MS = 500
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_EMBEDDING_DIM = 1536
DEFAULT_PROMPT_COLUMN = "prompt"

PROMPT_EMBEDDING_FIELD = "prompt_embedding"

TRANSIENT_STATUSES = {"429", "500", "503", "504", "resource_exhausted", "unavailable", "timeout"}

MAX_RETRIES = 6
INITIAL_BACKOFF_SECONDS = 5.0
MAX_BACKOFF_SECONDS = 300.0


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)


def err(msg: str) -> None:
    print(f"[{ts()}] {msg}", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add prompt_embedding field to a CSV by embedding the prompt column."
    )
    parser.add_argument("input_csv", help="Input CSV path.")
    parser.add_argument("--output", required=True, help="Output CSV path (also used for resume).")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Max new rows to process (0 = all).")
    parser.add_argument("--delay-ms", type=int, default=DEFAULT_DELAY_MS, help="Milliseconds to wait between rows.")
    parser.add_argument("--prompt-column", default=DEFAULT_PROMPT_COLUMN, help="Prompt column name. Default: prompt")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, help="Embedding model.")
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM, help="Embedding dimensionality.")
    return parser.parse_args()


def load_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY. Put it in a .env file or export it before running.")
    return api_key


def read_csv_rows(path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if reader.fieldnames is None:
            raise RuntimeError("CSV has no header.")
        return rows, list(reader.fieldnames)


def get_embedding_values(embed_response: Any) -> List[float]:
    embeddings = getattr(embed_response, "embeddings", None)
    if embeddings:
        values = getattr(embeddings[0], "values", None)
        if values is not None:
            return list(values)
    embedding = getattr(embed_response, "embedding", None)
    if embedding is not None:
        values = getattr(embedding, "values", None)
        if values is not None:
            return list(values)
    raise RuntimeError("Could not extract embedding values from response.")


def classify_exception(exc: Exception) -> str:
    msg = f"{type(exc).__name__}: {exc}".lower()
    for marker in TRANSIENT_STATUSES:
        if marker in msg:
            return "transient"
    return "fatal"


def with_backoff(fn, row_idx: int, stage: str):
    backoff = INITIAL_BACKOFF_SECONDS
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as exc:
            kind = classify_exception(exc)
            is_last = attempt == MAX_RETRIES
            if kind != "transient" or is_last:
                err(f"FAIL idx={row_idx} stage={stage} attempt={attempt} error={type(exc).__name__}: {exc}")
                raise
            jitter = random.uniform(0, 1.0)
            wait_s = min(backoff + jitter, MAX_BACKOFF_SECONDS)
            err(f"RETRY idx={row_idx} stage={stage} attempt={attempt}/{MAX_RETRIES} error={type(exc).__name__}: {exc} wait={wait_s:.1f}s")
            time.sleep(wait_s)
            backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)


def embed_text(client: genai.Client, model_name: str, text: str, output_dimensionality: int, row_idx: int) -> List[float]:
    def _call():
        response = client.models.embed_content(
            model=model_name,
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=output_dimensionality),
        )
        return get_embedding_values(response)
    return with_backoff(_call, row_idx=row_idx, stage="embed")


def write_all(output_path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_csv).resolve()
    output_path = Path(args.output).resolve()

    if not input_path.exists():
        err(f"Input file does not exist: {input_path}")
        return 1

    try:
        api_key = load_api_key()
    except Exception as exc:
        err(str(exc))
        return 1

    log("Initializing Gemini client")
    client = genai.Client(api_key=api_key)

    # Resume from output if it exists, otherwise start from input
    source_path = output_path if output_path.exists() else input_path
    try:
        rows, fieldnames = read_csv_rows(str(source_path))
    except Exception as exc:
        err(f"Read error: {type(exc).__name__}: {exc}")
        return 1

    if not rows:
        err("CSV is empty.")
        return 1

    if args.prompt_column not in fieldnames:
        err(f"Missing column '{args.prompt_column}' in {source_path}")
        return 1

    if PROMPT_EMBEDDING_FIELD not in fieldnames:
        fieldnames.append(PROMPT_EMBEDDING_FIELD)

    pending = [
        (i, row) for i, row in enumerate(rows, start=1)
        if not (row.get(PROMPT_EMBEDDING_FIELD) or "").strip()
    ]

    log(f"INPUT FILE: {input_path}")
    log(f"OUTPUT FILE: {output_path}")
    log(f"MODE: {'resume' if output_path.exists() else 'fresh'}")
    log(f"TOTAL ROWS: {len(rows)}")
    log(f"PENDING (no embedding): {len(pending)}")
    log(f"PROMPT COLUMN: {args.prompt_column}")
    log(f"EMBEDDING MODEL: {args.embedding_model}")
    log(f"EMBEDDING DIM: {args.embedding_dim}")
    log(f"DELAY BETWEEN ROWS: {args.delay_ms}ms")

    started = time.perf_counter()
    new_processed = 0

    try:
        for idx, row in pending:
            if args.limit > 0 and new_processed >= args.limit:
                log(f"LIMIT reached after {new_processed} new rows")
                break

            prompt = (row.get(args.prompt_column) or "").strip()
            log(f"START idx={idx}")

            if not prompt:
                err(f"SKIP idx={idx} reason=empty prompt")
                row[PROMPT_EMBEDDING_FIELD] = ""
                continue

            try:
                embedding = embed_text(
                    client=client,
                    model_name=args.embedding_model,
                    text=prompt,
                    output_dimensionality=args.embedding_dim,
                    row_idx=idx,
                )
                row[PROMPT_EMBEDDING_FIELD] = json.dumps(embedding, ensure_ascii=False)
                new_processed += 1
                log(f"DONE idx={idx}")

                if new_processed % 10 == 0:
                    elapsed = time.perf_counter() - started
                    log(f"processed {new_processed} prompts ({elapsed:.1f}s) — saving checkpoint")
                    write_all(output_path, fieldnames, rows)

                if args.delay_ms > 0:
                    time.sleep(args.delay_ms / 1000.0)

            except Exception as exc:
                kind = classify_exception(exc)
                msg = f"{type(exc).__name__}: {exc}"
                write_all(output_path, fieldnames, rows)
                if kind == "transient":
                    err(f"STOPPING_AFTER_TRANSIENT_FAILURE idx={idx} error={msg}")
                    err("Stopped cleanly after transient failure — re-run to resume.")
                    return 0
                else:
                    err(f"FATAL idx={idx} error={msg}")
                    return 1

    finally:
        write_all(output_path, fieldnames, rows)

    elapsed = time.perf_counter() - started
    log(f"FINISHED new_processed={new_processed} elapsed={elapsed:.1f}s output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
