#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import types

DEFAULT_LIMIT = 20
DEFAULT_DELAY_MS = 2000
DEFAULT_GENERATION_MODEL = "gemini-2.5-flash"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_EMBEDDING_DIM = 1536
DEFAULT_PROMPT_COLUMN = "prompt"

RESULT_FIELD = "resultado"
EMBEDDING_FIELD = "embedding"
PROMPT_EMBEDDING_FIELD = "prompt_embedding"
STATUS_FIELD = "status"
ERROR_FIELD = "error_message"
SOURCE_FIELD = "source_file"

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
        description="Generate Gemini outputs + embeddings from an input CSV, resuming into a fixed output CSV."
    )
    parser.add_argument("input_csv", help="Input CSV path. Must include a prompt column. Row IDs are derived via MD5 hash of the prompt.")
    parser.add_argument("--output", required=True, help="Output CSV path. This file is also used for resume/checkpointing.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="How many NEW rows to process in this run. Use 0 for all pending.")
    parser.add_argument("--delay-ms", type=int, default=DEFAULT_DELAY_MS, help="Milliseconds to wait between successfully processed rows.")
    parser.add_argument("--prompt-column", default=DEFAULT_PROMPT_COLUMN, help="Prompt column name. Default: prompt")
    parser.add_argument("--generation-model", default=DEFAULT_GENERATION_MODEL, help="Generation model.")
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
            raise RuntimeError("Input CSV has no header.")
        return rows, list(reader.fieldnames)

def ensure_required_columns(fieldnames: List[str], prompt_column: str) -> None:
    required = {prompt_column}
    missing = [c for c in required if c not in fieldnames]
    if missing:
        raise RuntimeError(f"Input CSV is missing required columns: {', '.join(missing)}")

def safe_extract_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    candidates = getattr(response, "candidates", None)
    if candidates:
        parts: List[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []) or []:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str):
                    parts.append(part_text)
        if parts:
            return "\n".join(parts).strip()

    return ""

def get_embedding_values(embed_response: Any) -> List[float]:
    embeddings = getattr(embed_response, "embeddings", None)
    if embeddings:
        first = embeddings[0]
        values = getattr(first, "values", None)
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
    if "api key" in msg or "permission" in msg or "unauthorized" in msg or "403" in msg:
        return "fatal"
    return "fatal"

def with_backoff(fn, row_id: str, stage: str):
    backoff = INITIAL_BACKOFF_SECONDS
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as exc:
            kind = classify_exception(exc)
            is_last = attempt == MAX_RETRIES
            if kind != "transient":
                err(f"FAIL id={row_id} stage={stage} kind=fatal attempt={attempt} error={type(exc).__name__}: {exc}")
                raise
            if is_last:
                err(f"FAIL id={row_id} stage={stage} kind=transient attempt={attempt} error={type(exc).__name__}: {exc}")
                raise

            jitter = random.uniform(0, 1.0)
            wait_s = min(backoff + jitter, MAX_BACKOFF_SECONDS)
            err(f"RETRY id={row_id} stage={stage} attempt={attempt}/{MAX_RETRIES} error={type(exc).__name__}: {exc} wait={wait_s:.1f}s")
            time.sleep(wait_s)
            backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)

def generate_text(client: genai.Client, model_name: str, prompt: str, row_id: str) -> str:
    def _call():
        response = client.models.generate_content(model=model_name, contents=prompt)
        return safe_extract_text(response)
    return with_backoff(_call, row_id=row_id, stage="generate")

def embed_text(client: genai.Client, model_name: str, text: str, output_dimensionality: int, row_id: str) -> List[float]:
    def _call():
        response = client.models.embed_content(
            model=model_name,
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=output_dimensionality),
        )
        return get_embedding_values(response)
    return with_backoff(_call, row_id=row_id, stage="embed")

def load_processed_ok_ids(output_path: Path) -> Set[str]:
    processed: Set[str] = set()
    if not output_path.exists():
        return processed
    with open(output_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_id = (row.get("id") or "").strip()
            status = (row.get(STATUS_FIELD) or "").strip().lower()
            if row_id and status == "ok":
                processed.add(row_id)
    return processed

def build_output_fieldnames(input_fieldnames: List[str]) -> List[str]:
    fieldnames = list(input_fieldnames)
    for extra in ("id", SOURCE_FIELD, RESULT_FIELD, EMBEDDING_FIELD, PROMPT_EMBEDDING_FIELD, STATUS_FIELD, ERROR_FIELD):
        if extra not in fieldnames:
            fieldnames.append(extra)
    return fieldnames

def open_output_writer(output_path: Path, fieldnames: List[str]):
    file_exists = output_path.exists()
    mode = "a" if file_exists else "w"
    f = open(output_path, mode, encoding="utf-8", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()
        f.flush()
    return f, writer, file_exists

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

    try:
        rows, input_fieldnames = read_csv_rows(str(input_path))
        ensure_required_columns(input_fieldnames, args.prompt_column)
    except Exception as exc:
        err(f"Input error: {type(exc).__name__}: {exc}")
        return 1

    if not rows:
        err("Input CSV is empty.")
        return 1

    output_fieldnames = build_output_fieldnames(input_fieldnames)
    processed_ok_ids = load_processed_ok_ids(output_path)

    try:
        out_f, writer, existed = open_output_writer(output_path, output_fieldnames)
    except Exception as exc:
        err(f"Could not open output file: {type(exc).__name__}: {exc}")
        return 1

    log(f"INPUT FILE: {input_path}")
    log(f"OUTPUT FILE: {output_path}")
    log(f"MODE: {'resume' if existed else 'fresh'}")
    log(f"ROWS IN INPUT: {len(rows)}")
    log(f"ALREADY PROCESSED (status=ok): {len(processed_ok_ids)}")
    log(f"PROMPT COLUMN: {args.prompt_column}")
    log(f"GENERATION MODEL: {args.generation_model}")
    log(f"EMBEDDING MODEL: {args.embedding_model}")
    log(f"EMBEDDING DIM: {args.embedding_dim}")
    log(f"DELAY BETWEEN ROWS: {args.delay_ms}ms")

    started = time.perf_counter()
    new_processed = 0

    try:
        for idx, row in enumerate(rows, start=1):
            prompt_raw = (row.get(args.prompt_column) or "")
            row_id = hashlib.md5(prompt_raw.encode("utf-8")).hexdigest()
            row["id"] = row_id

            if row_id in processed_ok_ids:
                continue

            if args.limit > 0 and new_processed >= args.limit:
                log(f"LIMIT reached after {new_processed} new rows")
                break

            prompt = prompt_raw.strip()
            log(f"START idx={idx} id={row_id}")

            if not prompt:
                row[SOURCE_FIELD] = input_path.name
                row[RESULT_FIELD] = ""
                row[EMBEDDING_FIELD] = ""
                row[PROMPT_EMBEDDING_FIELD] = ""
                row[STATUS_FIELD] = "bad_row"
                row[ERROR_FIELD] = "empty prompt"
                writer.writerow(row)
                out_f.flush()
                err(f"BAD_ROW idx={idx} id={row_id} reason=empty prompt")
                continue

            try:
                result_text = generate_text(client=client, model_name=args.generation_model, prompt=prompt, row_id=row_id)
                log(f"GEN OK id={row_id}")

                embedding = embed_text(
                    client=client,
                    model_name=args.embedding_model,
                    text=result_text,
                    output_dimensionality=args.embedding_dim,
                    row_id=row_id,
                )
                log(f"EMB OK id={row_id}")

                prompt_embedding = embed_text(
                    client=client,
                    model_name=args.embedding_model,
                    text=prompt,
                    output_dimensionality=args.embedding_dim,
                    row_id=row_id,
                )
                log(f"PEMB OK id={row_id}")

                row[SOURCE_FIELD] = input_path.name
                row[RESULT_FIELD] = result_text
                row[EMBEDDING_FIELD] = json.dumps(embedding, ensure_ascii=False)
                row[PROMPT_EMBEDDING_FIELD] = json.dumps(prompt_embedding, ensure_ascii=False)
                row[STATUS_FIELD] = "ok"
                row[ERROR_FIELD] = ""

                writer.writerow(row)
                out_f.flush()
                processed_ok_ids.add(row_id)
                new_processed += 1

                log(f"DONE idx={idx} id={row_id}")

                if new_processed % 10 == 0:
                    elapsed = time.perf_counter() - started
                    log(f"processed {new_processed} prompts ({elapsed:.1f} seconds)")

                if args.delay_ms > 0:
                    time.sleep(args.delay_ms / 1000.0)

            except Exception as exc:
                kind = classify_exception(exc)
                msg = f"{type(exc).__name__}: {exc}"

                row[SOURCE_FIELD] = input_path.name
                row[RESULT_FIELD] = ""
                row[EMBEDDING_FIELD] = ""
                row[PROMPT_EMBEDDING_FIELD] = ""
                row[STATUS_FIELD] = "transient_error" if kind == "transient" else "fatal_error"
                row[ERROR_FIELD] = msg
                writer.writerow(row)
                out_f.flush()

                if kind == "transient":
                    err(f"STOPPING_AFTER_TRANSIENT_FAILURE idx={idx} id={row_id} error={msg}")
                    err("Batch stopped cleanly after transient failure so it can be resumed later.")
                    break
                else:
                    err(f"FATAL idx={idx} id={row_id} error={msg}")
                    return 1

    finally:
        out_f.close()

    elapsed = time.perf_counter() - started
    log(f"FINISHED new_processed={new_processed} elapsed={elapsed:.1f}s output={output_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
