# Gemini Batch Scripts

Two scripts for processing CSV files with the Gemini or OpenAI APIs. Each script generates text and/or computes embeddings for every row in a CSV, with resume/checkpointing support so interrupted runs can be continued safely.

---

## Setup

### 1. Install dependencies

```bash
pip install google-genai python-dotenv
```

If you also want to use the OpenAI provider:

```bash
pip install openai
```

### 2. Configure API keys

Copy `.env.sample` to `.env` and fill in the keys you need:

```bash
cp .env.sample .env
```

---

## `run_gemini_batch_resume.py`

For each row in the input CSV, this script:
1. Generates a text response from the prompt
2. Embeds the generated response (`embedding`)
3. Embeds the original prompt (`prompt_embedding`)

Results are written row by row to the output CSV, which also serves as a checkpoint — re-running the same command skips rows already marked `status=ok`.

### Output columns added

| Column | Description |
|---|---|
| `id` | MD5 hash of the prompt (used for deduplication/resume) |
| `source_file` | Name of the input file |
| `resultado` | Generated text response |
| `embedding` | JSON array — embedding of the generated response |
| `prompt_embedding` | JSON array — embedding of the original prompt |
| `status` | `ok`, `bad_row`, `transient_error`, or `fatal_error` |
| `error_message` | Error detail when status is not `ok` |

### Usage

**Gemini (default):**

```bash
python run_gemini_batch_resume.py prompts.csv --output results.csv
```

**OpenAI:**

```bash
python run_gemini_batch_resume.py prompts.csv --output results.csv --provider openai
```

**Process 50 rows at a time, then re-run to continue:**

```bash
python run_gemini_batch_resume.py prompts.csv --output results.csv --limit 50
python run_gemini_batch_resume.py prompts.csv --output results.csv --limit 50  # resumes
```

**Process the entire file in one shot:**

```bash
python run_gemini_batch_resume.py prompts.csv --output results.csv --limit 0
```

**Override models explicitly:**

```bash
python run_gemini_batch_resume.py prompts.csv --output results.csv \
  --provider openai \
  --generation-model gpt-4o \
  --embedding-model text-embedding-3-large \
  --embedding-dim 3072
```

### Options

| Flag | Default (Gemini) | Default (OpenAI) | Description |
|---|---|---|---|
| `--output` | *(required)* | *(required)* | Output CSV path. Used for resume/checkpointing. |
| `--provider` | `gemini` | — | API provider: `gemini` or `openai`. |
| `--limit` | `20` | `20` | Max new rows to process per run. `0` = all pending. |
| `--delay-ms` | `2000` | `2000` | Milliseconds to wait between rows. |
| `--prompt-column` | `prompt` | `prompt` | Name of the column containing the prompts. |
| `--generation-model` | `gemini-2.5-flash` | `o4-mini` | Model used for text generation. |
| `--embedding-model` | `gemini-embedding-001` | `text-embedding-3-small` | Model used for embeddings. |
| `--embedding-dim` | `1536` | `1536` | Output dimensionality for embeddings. |

---

## `add_prompts.py`

Adds a `prompt_embedding` column to an existing CSV by embedding the prompt column for each row. No text generation — embeddings only. Useful for back-filling embeddings on a CSV produced without them.

Rows that already have a non-empty `prompt_embedding` are skipped. The entire file is rewritten on each run, with a checkpoint save every 10 rows.

This script has no `--provider` flag. Since it only calls one API (embeddings), passing `--embedding-model` with the desired model name is enough to switch providers.

### Usage

**Gemini (default):**

```bash
python add_prompts.py input.csv --output output.csv
```

**OpenAI:**

```bash
python add_prompts.py input.csv --output output.csv --embedding-model text-embedding-3-small
```

**Process in batches:**

```bash
python add_prompts.py input.csv --output output.csv --limit 100
python add_prompts.py input.csv --output output.csv --limit 100  # resumes
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--output` | *(required)* | Output CSV path. Used for resume/checkpointing. |
| `--limit` | `0` (all) | Max new rows to process per run. |
| `--delay-ms` | `500` | Milliseconds to wait between rows. |
| `--prompt-column` | `prompt` | Name of the column containing the prompts. |
| `--embedding-model` | `gemini-embedding-001` | Model used for embeddings. |
| `--embedding-dim` | `1536` | Output dimensionality for embeddings. |

---

## Resume behavior

Both scripts skip rows that have already been successfully processed. Re-running the same command is always safe.

- `run_gemini_batch_resume.py` — tracks processed rows by MD5 hash of the prompt; skips any row with `status=ok` in the output file.
- `add_prompts.py` — reads from the output file if it exists; skips any row with a non-empty `prompt_embedding`.

---

## Error handling

Both scripts use exponential backoff with jitter (up to 6 retries, capped at 5 minutes) for transient API errors (`429`, `500`, `503`, `504`, rate limit, timeout). On a transient failure the run stops cleanly so it can be resumed. Fatal errors (bad auth, invalid request) stop immediately with exit code `1`.
