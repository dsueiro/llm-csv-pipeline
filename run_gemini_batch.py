#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import types


DEFAULT_LIMIT = 20
DEFAULT_DELAY_MS = 0
DEFAULT_GENERATION_MODEL = "gemini-3-flash-preview"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_EMBEDDING_DIM = 1536
DEFAULT_PROMPT_COLUMN = "prompt"
DEFAULT_RESULT_COLUMN = "resultado"
DEFAULT_EMBEDDING_COLUMN = "embedding"

# Reintentos
MAX_RETRIES = 6
INITIAL_BACKOFF_SECONDS = 2.0
MAX_BACKOFF_SECONDS = 60.0


def log(msg: str) -> None:
    """Simple stdout logger with timestamp."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera respuestas y embeddings de Gemini a partir de un CSV."
    )
    parser.add_argument(
        "input_csv",
        help="Ruta al CSV de entrada. Debe incluir una columna 'prompt' salvo que uses --prompt-column.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Cantidad de requests a correr. Usa 0 para procesar todo. Default: {DEFAULT_LIMIT}",
    )
    parser.add_argument(
        "--delay-ms",
        type=int,
        default=DEFAULT_DELAY_MS,
        help=f"Milisegundos entre prompts. Default: {DEFAULT_DELAY_MS}",
    )
    parser.add_argument(
        "--prompt-column",
        default=DEFAULT_PROMPT_COLUMN,
        help=f"Nombre de la columna de prompts. Default: {DEFAULT_PROMPT_COLUMN}",
    )
    parser.add_argument(
        "--generation-model",
        default=DEFAULT_GENERATION_MODEL,
        help=f"Modelo de generación. Default: {DEFAULT_GENERATION_MODEL}",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Modelo de embeddings. Default: {DEFAULT_EMBEDDING_MODEL}",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=DEFAULT_EMBEDDING_DIM,
        help=f"Dimensionalidad del embedding. Default: {DEFAULT_EMBEDDING_DIM}",
    )
    return parser.parse_args()


def load_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No encontré GEMINI_API_KEY en el entorno. Definila en un .env."
        )
    return api_key


def read_csv_rows(path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if reader.fieldnames is None:
            raise RuntimeError("El CSV no tiene header.")
        return rows, list(reader.fieldnames)


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

    raise RuntimeError("No pude extraer el vector de embedding.")


def should_retry(exc: Exception) -> bool:
    msg = f"{type(exc).__name__}: {exc}".lower()
    retry_markers = [
        "429",
        "resource_exhausted",
        "rate limit",
        "quota",
        "503",
        "500",
        "timeout",
        "temporarily unavailable",
        "unavailable",
        "internal",
    ]
    return any(marker in msg for marker in retry_markers)


def with_backoff(fn, description: str):
    backoff = INITIAL_BACKOFF_SECONDS

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as exc:
            retry = should_retry(exc)
            is_last = attempt == MAX_RETRIES

            if not retry or is_last:
                log(
                    f"{description} failed on attempt {attempt}/{MAX_RETRIES}: "
                    f"{type(exc).__name__}: {exc}"
                )
                raise

            log(
                f"{description} failed on attempt {attempt}/{MAX_RETRIES}: "
                f"{type(exc).__name__}: {exc}. "
                f"Retrying in {backoff:.1f}s"
            )
            time.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)

    raise RuntimeError(f"{description} failed unexpectedly after retries.")


def generate_text(client: genai.Client, model_name: str, prompt: str) -> str:
    def _call():
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        return safe_extract_text(response)

    return with_backoff(_call, "generate_content")


def embed_text(
    client: genai.Client,
    model_name: str,
    text: str,
    output_dimensionality: int,
) -> List[float]:
    def _call():
        response = client.models.embed_content(
            model=model_name,
            contents=text,
            config=types.EmbedContentConfig(
                output_dimensionality=output_dimensionality
            ),
        )
        return get_embedding_values(response)

    return with_backoff(_call, "embed_content")


def make_output_path(input_csv: str) -> Path:
    timestamp = datetime.now().strftime("%d%m%y%H%M%S")
    parent = Path(input_csv).resolve().parent
    return parent / f"resultados_{timestamp}.csv"


def main() -> int:
    args = parse_args()

    try:
        api_key = load_api_key()
    except Exception as exc:
        log(str(exc))
        return 1

    log("Initializing Gemini client")
    client = genai.Client(api_key=api_key)

    try:
        rows, fieldnames = read_csv_rows(args.input_csv)
    except Exception as exc:
        log(f"Error reading CSV: {type(exc).__name__}: {exc}")
        return 1

    if not rows:
        log("The CSV is empty.")
        return 1

    prompt_column = args.prompt_column
    if prompt_column not in fieldnames:
        log(f"Column '{prompt_column}' not found in CSV.")
        return 1

    limit = args.limit if args.limit and args.limit > 0 else len(rows)
    delay_seconds = max(args.delay_ms, 0) / 1000.0
    output_path = make_output_path(args.input_csv)

    result_field = DEFAULT_RESULT_COLUMN
    embedding_field = DEFAULT_EMBEDDING_COLUMN

    if result_field not in fieldnames:
        fieldnames.append(result_field)
    if embedding_field not in fieldnames:
        fieldnames.append(embedding_field)

    log(f"Input file: {args.input_csv}")
    log(f"Output file: {output_path}")
    log(f"Rows in input: {len(rows)}")
    log(f"Rows to process: {limit}")
    log(f"Generation model: {args.generation_model}")
    log(f"Embedding model: {args.embedding_model}")
    log(f"Embedding dim: {args.embedding_dim}")
    log(f"Delay between prompts: {delay_seconds:.3f}s")

    started = time.perf_counter()
    processed = 0

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            if processed >= limit:
                break

            prompt = (row.get(prompt_column) or "").strip()

            if not prompt:
                row[result_field] = ""
                row[embedding_field] = ""
                writer.writerow(row)
                processed += 1
                continue

            try:
                result_text = generate_text(
                    client=client,
                    model_name=args.generation_model,
                    prompt=prompt,
                )

                embedding = embed_text(
                    client=client,
                    model_name=args.embedding_model,
                    text=result_text,
                    output_dimensionality=args.embedding_dim,
                )

                row[result_field] = result_text
                # JSON array en una sola celda: fácil de parsear luego
                row[embedding_field] = json.dumps(embedding, ensure_ascii=False)

            except Exception as exc:
                row[result_field] = f"[ERROR] {type(exc).__name__}: {exc}"
                row[embedding_field] = ""

            writer.writerow(row)
            processed += 1

            if processed % 10 == 0:
                elapsed = time.perf_counter() - started
                log(f"processed {processed} prompts ({elapsed:.1f} seconds)")

            if delay_seconds > 0 and processed < limit:
                time.sleep(delay_seconds)

    total_elapsed = time.perf_counter() - started
    log(f"Done. Processed {processed} prompts in {total_elapsed:.1f} seconds")
    log(f"Saved results to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
