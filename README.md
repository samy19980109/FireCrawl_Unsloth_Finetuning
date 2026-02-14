<div align="center">

# 🔥 Firecrawl → Unsloth Dataset Generator

### Turn any website into a fine-tuning dataset in one command

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Firecrawl](https://img.shields.io/badge/Firecrawl-Powered-FF6B35?style=for-the-badge&logo=fire&logoColor=white)](https://firecrawl.dev)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com)
[![Unsloth](https://img.shields.io/badge/Unsloth-Ready-4CAF50?style=for-the-badge&logo=rocket&logoColor=white)](https://github.com/unslothai/unsloth)

<br />

**Crawl** any website with Firecrawl &rarr; **Generate** high-quality Q&A pairs with GPT-4o-mini &rarr; **Export** ShareGPT + Alpaca JSONL &rarr; **Fine-tune** with Unsloth

<br />

</div>

---

## How It Works

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   🌐 Crawl   │────▶│  🧹 Clean &  │────▶│  🤖 Generate │────▶│  📦 Export   │
│   Firecrawl  │     │    Chunk     │     │   Q&A Pairs  │     │   Datasets   │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
   Up to 250           Markdown cleaning     Async OpenAI        ShareGPT JSONL
   pages per run       3-tier chunking       Structured output   Alpaca JSONL
                       Boilerplate removal   Multi-turn convos   Stats JSON
```

> **One command. Hundreds of pages. Thousands of Q&A pairs. Ready for Unsloth.**

---

## Quick Start

```bash
# 1. Clone & install
git clone <repo-url> && cd firecrawl_unsloth_finetuning
pip install -r requirements.txt

# 2. Add your API keys
cp .env.example .env
# Edit .env → add FIRECRAWL_API_KEY and OPENAI_API_KEY

# 3. Generate a dataset
python main.py https://docs.example.com
```

That's it. Your dataset files will be in `./output/`.

---

## Usage

```bash
# Crawl a docs site with defaults
python main.py https://docs.example.com

# Full control over the pipeline
python main.py https://docs.example.com \
  --max-pages 100 \
  --qa-per-chunk 5 \
  --chunk-size 2000 \
  --concurrency 20 \
  --model gpt-4o-mini \
  --output-dir ./my-dataset

# Include multi-turn conversations (user → assistant → user → assistant)
python main.py https://docs.example.com --multi-turn
```

### All Options

| Flag | Default | Description |
|:-----|:--------|:------------|
| `url` | *(required)* | Starting URL to crawl |
| `--max-pages` | `250` | Maximum pages to crawl |
| `--output-dir` | `./output` | Directory for output files |
| `--chunk-size` | `3000` | Max characters per text chunk |
| `--qa-per-chunk` | `2` | Q&A pairs to generate per chunk |
| `--model` | `gpt-4o-mini` | OpenAI model for Q&A generation |
| `--concurrency` | `10` | Parallel OpenAI requests |
| `--dedup-threshold` | `85` | Fuzzy similarity threshold for dedup (0–100) |
| `--multi-turn` | off | Also generate 4-turn conversations |

---

## Pipeline Deep Dive

### 1. Crawl

Uses the [Firecrawl SDK](https://firecrawl.dev) to crawl the target URL, extracting clean markdown with `only_main_content=True`. Pages with fewer than 100 characters of content are automatically discarded.

### 2. Clean & Chunk

Strips boilerplate (nav menus, cookie banners, "skip to content" links, image-only lines) then splits content using a **three-tier chunking strategy**:

| Tier | Strategy | Purpose |
|:-----|:---------|:--------|
| 1 | Split on `#` / `##` / `###` headers | Respect document structure |
| 2 | Merge small sections, split large ones by paragraph | Hit target chunk size |
| 3 | Sliding window with overlap | Handle oversized paragraphs |

### 3. Generate Q&A

Fires async requests to OpenAI with **structured output** (Pydantic models), generating:

- **Single-turn Q&A** — Diverse question types: factual, how-to, conceptual, comparison, troubleshooting
- **Multi-turn conversations** (optional) — 4-turn dialogues where follow-ups build naturally on prior answers

Quality is baked into the prompts: answers must be 4–8 sentences with specific details, questions must be self-contained, and meta-references ("the text says...") are forbidden.

### 4. Filter & Deduplicate

Before export, the pipeline runs two passes:

- **Quality filter** — Removes short answers (< 30 chars), short questions (< 15 chars), and any remaining meta-references
- **Fuzzy dedup** — Uses [RapidFuzz](https://github.com/maxbachmann/RapidFuzz) token sort ratio to eliminate near-duplicate questions

### 5. Export

Writes dataset files ready for Unsloth:

---

## Output Formats

<table>
<tr>
<td><strong>ShareGPT</strong> — <code>dataset_sharegpt.jsonl</code></td>
<td><strong>Alpaca</strong> — <code>dataset_alpaca.jsonl</code></td>
</tr>
<tr>
<td>

```json
{
  "conversations": [
    {"from": "human", "value": "What is..."},
    {"from": "gpt", "value": "Detailed answer..."}
  ]
}
```

</td>
<td>

```json
{
  "instruction": "What is...",
  "input": "",
  "output": "Detailed answer..."
}
```

</td>
</tr>
</table>

When `--multi-turn` is enabled, an additional **`dataset_sharegpt_multi_turn.jsonl`** is produced with 4-turn conversations.

A **`stats.json`** file is also generated with full run statistics (pages scraped, chunks processed, pairs generated, pairs filtered, etc.).

---

## Fault Tolerance

The pipeline **checkpoints every chunk** to `<output-dir>/.checkpoint.jsonl`. If a run crashes or is interrupted:

```bash
# Just re-run the same command — it picks up where it left off
python main.py https://docs.example.com --output-dir ./output
```

The checkpoint is automatically cleaned up after a successful run.

---

## Architecture

```
main.py                     # Entire pipeline in one file
├── crawl_website()         # Firecrawl SDK integration
├── clean_markdown()        # Boilerplate stripping
├── chunk_text()            # 3-tier text splitting
├── generate_all()          # Async orchestrator with checkpointing
│   ├── generate_qa_pairs()         # Single-turn Q&A via structured output
│   └── generate_multi_turn()       # Multi-turn conversations
├── quality_filter()        # Min-length + meta-reference removal
├── deduplicate()           # RapidFuzz near-duplicate removal
├── to_sharegpt() / to_alpaca()    # Format converters
└── run_pipeline()          # End-to-end orchestration
```

---

## Requirements

- Python 3.10+
- [Firecrawl API key](https://firecrawl.dev) — for web crawling
- [OpenAI API key](https://platform.openai.com) — for Q&A generation

```
firecrawl
openai
python-dotenv
tqdm
pydantic
rapidfuzz
```

---

<div align="center">
<br />

Built with [Firecrawl](https://firecrawl.dev) and [OpenAI](https://openai.com) — dataset-ready for [Unsloth](https://github.com/unslothai/unsloth) fine-tuning

</div>
