# Firecrawl to Unsloth Fine-Tuning Dataset

Crawl any website with [Firecrawl](https://firecrawl.dev), generate Q&A pairs using OpenAI GPT-4o-mini, and export datasets in **ShareGPT** and **Alpaca** JSONL formats ready for [Unsloth](https://github.com/unslothai/unsloth) QLoRA/LoRA fine-tuning.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your real keys
```

## Usage

```bash
# Basic usage
python main.py https://docs.example.com

# With options
python main.py https://docs.example.com \
  --max-pages 20 \
  --qa-per-chunk 5 \
  --chunk-size 1500 \
  --output-dir ./output \
  --model gpt-4o-mini

# Generate multi-turn conversations alongside single-turn Q&A
python main.py https://docs.example.com --multi-turn
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `url` | *(required)* | Starting URL to crawl |
| `--max-pages` | 250 | Maximum pages to crawl |
| `--output-dir` | `./output` | Directory for output files |
| `--chunk-size` | 3000 | Max characters per text chunk |
| `--qa-per-chunk` | 2 | Q&A pairs to generate per chunk |
| `--model` | `gpt-4o-mini` | OpenAI model for Q&A generation |
| `--concurrency` | 10 | Max concurrent OpenAI API requests |
| `--dedup-threshold` | 85 | Fuzzy dedup similarity threshold (0-100) |
| `--multi-turn` | off | Also generate multi-turn conversations for ShareGPT |

### Checkpointing

The pipeline saves progress to `<output-dir>/.checkpoint.jsonl` as chunks are processed. If a run is interrupted, re-running the same command will resume from where it left off. The checkpoint file is automatically deleted on successful completion.

## Output

The tool produces these files in the output directory:

- **`dataset_sharegpt.jsonl`** — Single-turn conversations in ShareGPT format
- **`dataset_alpaca.jsonl`** — Single-turn instructions in Alpaca format
- **`dataset_sharegpt_multi_turn.jsonl`** — Multi-turn conversations in ShareGPT format (only when `--multi-turn` is used)
- **`stats.json`** — Summary statistics for the run

### ShareGPT format

```json
{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
```

### Alpaca format

```json
{"instruction": "...", "input": "", "output": "..."}
```
