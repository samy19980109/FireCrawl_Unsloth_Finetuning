# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Single-file Python CLI tool that crawls websites using Firecrawl, generates Q&A training datasets using OpenAI, and exports them in ShareGPT and Alpaca JSONL formats for Unsloth QLoRA/LoRA fine-tuning.

## Commands

```bash
# Setup
python -m venv venv313
source venv313/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Then add real FIRECRAWL_API_KEY and OPENAI_API_KEY

# Run
python main.py <url>
python main.py <url> --max-pages 20 --qa-per-chunk 5 --chunk-size 1500 --output-dir ./output --model gpt-4o-mini
python main.py <url> --multi-turn  # Also generate multi-turn conversations
python main.py <url> --concurrency 10 --dedup-threshold 85
```

No test suite or linter is configured.

## Architecture

Everything lives in `main.py`. The pipeline runs sequentially through these stages:

1. **Crawl** (`crawl_website`) — Uses Firecrawl SDK to crawl a URL, returns pages with markdown content (filters out pages < 100 chars)
2. **Clean** (`clean_markdown`) — Strips nav menus, image-only lines, cookie/boilerplate text, collapse blank lines
3. **Chunk** (`chunk_text`) — Three-tier splitting: markdown headers → paragraph merging → sliding window with overlap. Min chunk size 50 chars
4. **Generate** (`generate_all`) — Async OpenAI calls with semaphore-based concurrency control. Generates both single-turn Q&A (`QAResponse`) and optional multi-turn conversations (`MultiTurnResponse`) using Pydantic structured output parsing. Includes JSONL checkpointing for resume on failure
5. **Filter** (`quality_filter`) — Removes short answers (< 30 chars), short questions (< 15 chars), and meta-references ("the text says...")
6. **Deduplicate** (`deduplicate`) — Uses rapidfuzz `token_sort_ratio` to remove near-duplicate questions above a similarity threshold
7. **Export** — Writes `dataset_sharegpt.jsonl`, `dataset_alpaca.jsonl`, optional `dataset_sharegpt_multi_turn.jsonl`, and `stats.json`. Cleans up checkpoint file on success

## Key Design Decisions

- **Checkpointing**: Results are saved incrementally to `<output_dir>/.checkpoint.jsonl` keyed by deterministic SHA-256 chunk IDs. The pipeline resumes from checkpoint on restart and deletes it on successful completion.
- **Structured output**: Uses `client.chat.completions.parse()` with Pydantic models (`QAResponse`, `MultiTurnResponse`) rather than free-form JSON parsing.
- **Concurrency**: Async with `asyncio.Semaphore` (default 10) for rate-limiting OpenAI API calls. Retries up to 3 times with exponential backoff.
- **Quality prompts**: System prompts enforce 4-8 sentence answers, self-contained questions, no meta-references, and question type variety.

## Environment Variables

Loaded from `.env` via `python-dotenv`:
- `FIRECRAWL_API_KEY` — Required for website crawling
- `OPENAI_API_KEY` — Required for Q&A generation
