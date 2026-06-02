# nimbench

`nimbench` is a small CLI for measuring response latency across NVIDIA NIM chat models.

It does three things:

1. Fetches available models from `GET /v1/models`
2. Selects likely chat-capable models by default
3. Sends a tiny chat completion request to each candidate model
4. Sorts results by fastest median response time

The app is intentionally lightweight. It uses only Python stdlib and writes progress logs while it runs.

## What it measures

`nimbench` measures wall-clock request time for a minimal `POST /v1/chat/completions` call.

Default request shape:

- prompt: `Reply with one short word.`
- `max_tokens: 8`
- `temperature: 0`, with automatic fallback to `0.1` if a backend rejects zero temperature

That makes the benchmark mostly about request/response latency, not output quality or throughput.

The CLI also reports tokens per second for each successful model. If NVIDIA returns per-request metrics, the server value is used. Otherwise the app derives an approximate rate from `completion_tokens / wall_time`.

## Install

Requires Python 3.10+.

```bash
python3 -m pip install -e .
```

## Run

Set your NVIDIA API key in one of these ways:

```bash
nimbench --api-key nvapi-...
```

or:

```bash
export NVIDIA_API_KEY=nvapi-...
nimbench
```

If no key is passed, the app prompts for one.

Basic run:

```bash
python3 -m nimbench --limit 10
```

Useful example:

```bash
python3 -m nimbench \
  --api-key nvapi-... \
  --limit 10 \
  --pattern 'llama|nemotron|gpt-oss|qwen|mistral'
```

## How it behaves

- Models are discovered from `GET /v1/models`
- Chat-like model ids are selected by default
- `--all-models` restores full catalog benchmarking
- Benchmarking is sequential
- The tool enforces a fixed `40 rpm` request cap
- The tool keeps a local skip cache for models that are not provisioned, reject chat input, or repeatedly time out
- `--refresh-cache` ignores that cache for one run and rebuilds it from fresh results
- Logs go to stderr
- Final results go to stdout

Important detail: `--limit` means "stop after N successful benchmarks", not "take the first N discovered rows". That avoids wasting the cap on models that are not available for chat completions.

## Options

```text
--api-key KEY      NVIDIA API key
--base-url URL     API base URL
--limit N          Stop after N successful benchmarks
--pattern REGEX    Only consider model ids matching REGEX
--timeout SECONDS  Request timeout for each HTTP call
--repeats N        Requests per model
--json             Emit JSON instead of a text table
--rpm N            Request rate cap, default 40
--concurrency N    Reserved. Benchmarking is sequential to preserve the cap.
--all-models       Benchmark full catalog instead of chat-only default
--refresh-cache    Ignore the local skip cache for this run
```

Default base URL:

```text
https://integrate.api.nvidia.com/v1
```

## Output

Text output contains:

- selected / attempted / discovered counts
- skipped non-chat and cached counts when applicable
- sorted successful models with median, min, max latency
- failed or unavailable models with error text

Example shape:

```text
NIM bench: 10 candidate model(s), 14 attempted, 117 discovered at https://integrate.api.nvidia.com/v1

rank  model                        median ms  min ms  max ms  tok/s  ok  err
...
```

Use `--json` if you want machine-readable output.

## Why some models fail

The NVIDIA catalog can expose more than chat-capable LLMs. Some entries are embeddings, safety, vision, or other task-specific models. Those may not answer `POST /v1/chat/completions` and can return `404`, timeout, or similar errors.

`nimbench` logs those as unavailable or failed and keeps going.

Models that repeatedly fail for the same reason are written to a local skip cache under the user cache directory. Later runs skip those models before sending a request, which keeps the benchmark focused on models that are actually usable for this API key. Use `--refresh-cache` to ignore the cache for one run.
Set `NIMBENCH_CACHE_DIR` if you want that cache in a different location.

## Development

Run tests:

```bash
python3 -m unittest discover -s tests -v
```

The package entrypoint is `nimbench`, and `python3 -m nimbench --help` shows the CLI help.
