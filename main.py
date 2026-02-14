"""
Firecrawl Website Scraper → Unsloth Fine-Tuning Dataset Generator

Crawls a website using Firecrawl, generates Q&A pairs with OpenAI,
and exports ShareGPT + Alpaca JSONL datasets for Unsloth fine-tuning.
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
from tqdm.asyncio import tqdm as atqdm

from rapidfuzz import fuzz

from firecrawl import Firecrawl
from firecrawl.types import ScrapeOptions

load_dotenv()

# ---------------------------------------------------------------------------
# Pydantic models for OpenAI structured output
# ---------------------------------------------------------------------------

class QAPair(BaseModel):
    question: str
    answer: str


class QAResponse(BaseModel):
    qa_pairs: List[QAPair]


class ConversationTurn(BaseModel):
    role: str  # "human" or "gpt"
    content: str


class MultiTurnConversation(BaseModel):
    turns: List[ConversationTurn]


class MultiTurnResponse(BaseModel):
    conversations: List[MultiTurnConversation]


# ---------------------------------------------------------------------------
# Crawling
# ---------------------------------------------------------------------------

def crawl_website(url: str, max_pages: int) -> list[dict]:
    """Crawl *url* with Firecrawl and return a list of page dicts."""
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        raise RuntimeError("FIRECRAWL_API_KEY not set – check your .env file")

    client = Firecrawl(api_key=api_key)
    print(f"Crawling {url} (max {max_pages} pages)…")

    result = client.crawl(
        url,
        limit=max_pages,
        scrape_options=ScrapeOptions(formats=["markdown"], only_main_content=True),
        poll_interval=5,
    )

    print(f"  Crawl status: {result.status} — {result.completed}/{result.total} pages")

    pages = []
    for doc in result.data:
        markdown = doc.markdown or ""
        if len(markdown) < 100:
            continue
        meta = doc.metadata
        source_url = (meta.source_url or meta.url or url) if meta else url
        title = (meta.title or "") if meta else ""
        pages.append({
            "markdown": markdown,
            "source_url": source_url,
            "title": title,
        })

    print(f"  → {len(pages)} usable pages (≥100 chars)")
    return pages


# ---------------------------------------------------------------------------
# Content cleaning
# ---------------------------------------------------------------------------

def clean_markdown(text: str) -> str:
    """Strip common website boilerplate from markdown before chunking."""
    # Remove blocks of consecutive markdown links (nav menus, footers)
    # e.g. lines that are just "[Link text](url)" repeated
    text = re.sub(
        r"(?:^[ \t]*\[.*?\]\(.*?\)[ \t]*\n){4,}",
        "\n",
        text,
        flags=re.MULTILINE,
    )
    # Remove image-only lines (decorative banners, icons)
    text = re.sub(r"^!\[.*?\]\(.*?\)\s*$", "", text, flags=re.MULTILINE)
    # Remove cookie/consent/subscription boilerplate phrases
    text = re.sub(
        r"^.*?(cookie|subscribe to our|sign up for|newsletter|accept all|privacy policy|terms of service).*$",
        "",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    # Remove "Skip to content", "Back to top", etc.
    text = re.sub(
        r"^.*?(skip to|back to top|jump to|table of contents).*$",
        "",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> list[str]:
    """Split *text* into chunks using a three-tier strategy."""
    # Tier 1 – split on markdown headers
    sections = re.split(r"(?=^#{1,3}\s)", text, flags=re.MULTILINE)
    sections = [s.strip() for s in sections if s.strip()]

    # Tier 2 – merge small sections, split large ones by paragraphs
    merged: list[str] = []
    buffer = ""
    for section in sections:
        if len(buffer) + len(section) <= chunk_size:
            buffer = f"{buffer}\n\n{section}" if buffer else section
        else:
            if buffer:
                merged.append(buffer)
            if len(section) <= chunk_size:
                buffer = section
            else:
                # Split oversized section by paragraphs
                paragraphs = section.split("\n\n")
                for para in paragraphs:
                    if len(buffer) + len(para) <= chunk_size:
                        buffer = f"{buffer}\n\n{para}" if buffer else para
                    else:
                        if buffer:
                            merged.append(buffer)
                        buffer = para
    if buffer:
        merged.append(buffer)

    # Tier 3 – sliding-window for anything still too large
    final: list[str] = []
    for chunk in merged:
        if len(chunk) <= chunk_size:
            final.append(chunk)
        else:
            start = 0
            while start < len(chunk):
                end = start + chunk_size
                final.append(chunk[start:end])
                start += chunk_size - overlap

    # Filter out tiny chunks
    return [c for c in final if len(c) >= 50]


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def chunk_id(chunk: str, source_url: str) -> str:
    """Deterministic ID for a (chunk, source_url) pair."""
    return hashlib.sha256(f"{source_url}|{chunk}".encode()).hexdigest()[:16]


def load_checkpoint(checkpoint_path: Path) -> dict[str, dict]:
    """Load already-generated results keyed by chunk ID."""
    completed: dict[str, dict] = {}
    if not checkpoint_path.exists():
        return completed
    with open(checkpoint_path, "r", encoding="utf-8") as fh:
        for line in fh:
            entry = json.loads(line)
            # Support both old format (qa_pairs list) and new format (data dict)
            if "data" in entry:
                completed[entry["chunk_id"]] = entry["data"]
            elif "qa_pairs" in entry:
                completed[entry["chunk_id"]] = entry["qa_pairs"]
    return completed


def append_checkpoint(checkpoint_path: Path, cid: str, data) -> None:
    """Append one chunk's results to the checkpoint file."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"chunk_id": cid, "data": data}, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Q&A generation (async)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert at reading website content and generating high-quality, \
diverse question-answer pairs for training a language model.

Given a passage of text, generate Q&A pairs following these rules:

1. VARIETY — each pair must use a DIFFERENT question type. Rotate through:
   - Factual: "What is…", "How many…", "When…"
   - How-to: "How do you…", "What are the steps to…"
   - Conceptual: "Why does…", "What is the purpose of…"
   - Comparison: "What is the difference between…", "How does X compare to…"
   - Troubleshooting: "What happens if…", "How do you fix…"

2. SELF-CONTAINED — each question must make sense without seeing the source \
text. Include enough context in the question itself (e.g. name the company, \
product, feature, or concept).

3. DETAILED ANSWERS — answers MUST be 4-8 sentences long and include specific \
details like product names, technical specs, numbers, locations, and reasoning. \
A one-sentence answer is NEVER acceptable. Explain the "why" and "how", not \
just the "what".

4. NO META-REFERENCES — never say "the text says", "according to the \
passage", "the documentation mentions", or similar.

## Examples of GOOD Q&A pairs (follow this level of detail)

Q: What warranty does Reiz Electrocontrols offer on their LED luminaires, and \
what does it cover?
A: Reiz Electrocontrols provides a European standard 5-year warranty on all \
their LED luminaires. This warranty covers the complete luminaire as an \
integrated unit, including the driver, LED module, and fixture, rather than \
covering individual components separately. The company is able to offer this \
comprehensive warranty because they use a vertically integrated manufacturing \
approach, controlling quality at every stage of production. Their drivers and \
LED luminaires also carry B.I.S., UL, and CE certifications, which reflects \
the high manufacturing standards that back the warranty. This level of coverage \
is particularly important for commercial installations in hotels, showrooms, \
and retail spaces where replacement costs and downtime are significant concerns.

Q: Why is a high Color Rendering Index important for LED lighting, and what CRI \
range does Reiz Electrocontrols offer?
A: Reiz Electrocontrols offers LED lights with a Color Rendering Index (CRI) of \
85/95, which is well above the typical commercial LED range of 70-80. A high CRI \
is important because it determines how accurately a light source renders the true \
colors of objects compared to natural sunlight, which has a perfect CRI of 100. \
With a CRI of 95, colors appear vivid and natural rather than washed out or \
shifted, which is why these lights are ideal for environments where color accuracy \
matters — such as museums, art galleries, photography studios, and retail \
showrooms displaying fabrics or jewellery. The difference between CRI 80 and CRI \
95 is clearly visible to the human eye, especially in reds and skin tones. Higher \
CRI lighting also reduces eye strain during extended exposure, making it a better \
choice for schools and office environments.

## Examples of BAD answers (too short — NEVER do this)

BAD: "The warranty is 5 years."
BAD: "Reiz Electrocontrols offers lights for hotels."
BAD: "The CRI range is 85/95."
\
"""

MULTI_TURN_SYSTEM_PROMPT = """\
You are an expert at reading website content and generating realistic multi-turn \
conversations for training a chat language model.

Given a passage of text, generate conversations where a user asks a question, \
the assistant answers in detail, and the user follows up with a naturally related \
deeper question. Each conversation must have exactly 4 turns: user, assistant, \
user, assistant.

Rules:
1. Each conversation must focus on a DIFFERENT topic from the passage.
2. The follow-up question should naturally build on the first answer — asking \
for more detail, a practical example, an edge case, or a comparison.
3. All questions must be SELF-CONTAINED — include enough context (company name, \
product, feature) to make sense on their own.
4. Every answer MUST be 4-8 sentences with specific details. Never give a \
one-sentence answer.
5. NO META-REFERENCES — never say "the text says", "the passage mentions", etc.
6. Turns must strictly alternate: human, gpt, human, gpt.

## Example

Turn 1 (human): What types of underwater LED lighting does Reiz Electrocontrols \
offer, and where can they be used?
Turn 2 (gpt): Reiz Electrocontrols offers underwater LED lights with an IP68 \
rating, which is the highest level of ingress protection available. This rating \
means the lights are fully sealed against dust and can operate continuously \
while submerged in water up to 3 meters deep. They are designed for use in \
swimming pools, decorative fountains, water features, and other aquatic \
installations where reliable, long-lasting illumination is needed. The lights \
provide vibrant, even illumination that enhances the visual appeal of the water \
environment while maintaining the same flicker-free and glare-reducing \
characteristics found across the Reiz Electrocontrols product line.
Turn 3 (human): How do the wall grazer LED lights from Reiz Electrocontrols \
differ from the underwater lights in terms of their design goals?
Turn 4 (gpt): While the underwater lights are engineered primarily for \
submersion durability with their IP68 sealing, the outdoor wall grazer LED \
lights are designed with a focus on broad, uniform light distribution across \
large vertical surfaces. The wall grazers provide even coverage without patches \
or hotspots, making them ideal for highlighting building facades, textured \
walls, and architectural details from the ground up. They also incorporate \
glare-reduction technology so they don't create discomfort for people looking \
toward the illuminated surface. Both product lines share Reiz's core commitment \
to visual comfort and energy efficiency, but the wall grazers prioritize \
aesthetic illumination of architecture while the underwater lights prioritize \
waterproof endurance and vibrant color output in aquatic settings.\
"""


async def _call_openai_structured(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    messages: list[dict],
    response_format,
    model: str,
):
    """Shared helper: call OpenAI with structured output, retries, and semaphore."""
    async with semaphore:
        for attempt in range(3):
            try:
                completion = await client.chat.completions.parse(
                    model=model,
                    messages=messages,
                    response_format=response_format,
                )
                message = completion.choices[0].message
                if message.parsed:
                    return message.parsed
                return None
            except Exception as exc:
                wait = 2 ** attempt
                print(f"  ⚠ Attempt {attempt + 1} failed ({exc}), retrying in {wait}s…")
                await asyncio.sleep(wait)
    return None


async def generate_qa_pairs(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    chunk: str,
    source_url: str,
    num_pairs: int = 3,
    model: str = "gpt-4o-mini",
) -> list[dict]:
    """Generate single-turn Q&A pairs from a chunk."""
    user_msg = (
        f"Generate {num_pairs} question-answer pairs from the following text.\n\n"
        f"Source: {source_url}\n\n"
        f"---\n{chunk}\n---"
    )
    parsed = await _call_openai_structured(
        client, semaphore,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format=QAResponse,
        model=model,
    )
    if parsed:
        return [
            {"question": qa.question, "answer": qa.answer, "source_url": source_url}
            for qa in parsed.qa_pairs
        ]
    return []


async def generate_multi_turn(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    chunk: str,
    source_url: str,
    num_convos: int = 1,
    model: str = "gpt-4o-mini",
) -> list[dict]:
    """Generate multi-turn conversations from a chunk."""
    user_msg = (
        f"Generate {num_convos} multi-turn conversation(s) (4 turns each) "
        f"from the following text.\n\n"
        f"Source: {source_url}\n\n"
        f"---\n{chunk}\n---"
    )
    parsed = await _call_openai_structured(
        client, semaphore,
        messages=[
            {"role": "system", "content": MULTI_TURN_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format=MultiTurnResponse,
        model=model,
    )
    if parsed:
        results = []
        for convo in parsed.conversations:
            turns = [{"from": t.role, "value": t.content} for t in convo.turns]
            if len(turns) >= 4:
                results.append({"turns": turns, "source_url": source_url})
        return results
    return []


async def generate_all(
    chunks: list[tuple[str, str]],
    checkpoint_path: Path,
    num_pairs: int,
    model: str,
    concurrency: int,
    multi_turn: bool,
) -> tuple[list[dict], list[dict]]:
    """Generate Q&A + multi-turn for all chunks with concurrency and checkpointing.

    Returns (single_turn_qa_pairs, multi_turn_conversations).
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY not set – check your .env file")

    client = AsyncOpenAI(api_key=openai_key)
    semaphore = asyncio.Semaphore(concurrency)
    completed = load_checkpoint(checkpoint_path)

    # Split into already-done vs. pending
    pending: list[tuple[int, str, str, str]] = []
    for i, (chunk, source_url) in enumerate(chunks):
        cid = chunk_id(chunk, source_url)
        if cid not in completed:
            pending.append((i, cid, chunk, source_url))

    if completed:
        print(f"  Resuming: {len(completed)} chunks already done, {len(pending)} remaining")
    else:
        print(f"  Processing {len(pending)} chunks (concurrency={concurrency})")

    pbar = atqdm(total=len(pending), desc="Generating Q&A")

    async def process_chunk(cid: str, chunk: str, source_url: str) -> tuple[str, dict]:
        qa_pairs = await generate_qa_pairs(client, semaphore, chunk, source_url, num_pairs, model)
        mt_convos = []
        if multi_turn:
            mt_convos = await generate_multi_turn(client, semaphore, chunk, source_url, 1, model)
        result = {"qa_pairs": qa_pairs, "multi_turn": mt_convos}
        append_checkpoint(checkpoint_path, cid, result)
        pbar.update(1)
        return cid, result

    tasks = [process_chunk(cid, chunk, src) for _, cid, chunk, src in pending]
    results = await asyncio.gather(*tasks)
    pbar.close()

    for cid, result in results:
        completed[cid] = result

    # Collect all results in original chunk order
    all_qa: list[dict] = []
    all_mt: list[dict] = []
    for chunk, source_url in chunks:
        cid = chunk_id(chunk, source_url)
        entry = completed.get(cid, {})
        # Handle legacy checkpoint format (list of qa_pairs directly)
        if isinstance(entry, list):
            all_qa.extend(entry)
        else:
            all_qa.extend(entry.get("qa_pairs", []))
            all_mt.extend(entry.get("multi_turn", []))

    return all_qa, all_mt


# ---------------------------------------------------------------------------
# Quality filtering & deduplication
# ---------------------------------------------------------------------------

META_PHRASES = re.compile(
    r"(?:the (?:text|passage|article|document|excerpt|section|paragraph) "
    r"(?:says|states|mentions|describes|explains|indicates|notes|suggests|provides|discusses))"
    r"|(?:according to the (?:text|passage|article|document|excerpt))"
    r"|(?:as (?:stated|mentioned|described|noted|explained) in the)",
    re.IGNORECASE,
)

MIN_ANSWER_LENGTH = 30
MIN_QUESTION_LENGTH = 15


def quality_filter(qa_pairs: list[dict]) -> tuple[list[dict], dict]:
    """Remove low-quality Q&A pairs. Returns (kept, removal_stats)."""
    stats = {"short_answer": 0, "short_question": 0, "meta_reference": 0}
    kept = []

    for qa in qa_pairs:
        q, a = qa["question"].strip(), qa["answer"].strip()

        if len(a) < MIN_ANSWER_LENGTH:
            stats["short_answer"] += 1
            continue

        if len(q) < MIN_QUESTION_LENGTH:
            stats["short_question"] += 1
            continue

        if META_PHRASES.search(q) or META_PHRASES.search(a):
            stats["meta_reference"] += 1
            continue

        kept.append(qa)

    return kept, stats


def deduplicate(qa_pairs: list[dict], threshold: int = 85) -> tuple[list[dict], int]:
    """Remove near-duplicate questions using rapidfuzz. Returns (deduped, num_removed)."""
    kept: list[dict] = []
    kept_questions: list[str] = []
    removed = 0

    for qa in qa_pairs:
        q = qa["question"]
        is_dup = False
        for existing_q in kept_questions:
            if fuzz.token_sort_ratio(q, existing_q) >= threshold:
                is_dup = True
                break
        if is_dup:
            removed += 1
        else:
            kept.append(qa)
            kept_questions.append(q)

    return kept, removed


# ---------------------------------------------------------------------------
# Format converters
# ---------------------------------------------------------------------------

def to_sharegpt(qa: dict) -> dict:
    return {
        "conversations": [
            {"from": "human", "value": qa["question"]},
            {"from": "gpt", "value": qa["answer"]},
        ],
        "source_url": qa.get("source_url", ""),
    }


def to_sharegpt_multi_turn(convo: dict) -> dict:
    return {
        "conversations": convo["turns"],
        "source_url": convo.get("source_url", ""),
    }


def to_alpaca(qa: dict) -> dict:
    return {
        "instruction": qa["question"],
        "input": "",
        "output": qa["answer"],
        "source_url": qa.get("source_url", ""),
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_jsonl(data: list[dict], filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as fh:
        for item in data:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  → Saved {len(data)} entries to {filepath}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

async def run_pipeline(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / ".checkpoint.jsonl"

    # 1. Crawl
    pages = crawl_website(args.url, args.max_pages)
    if not pages:
        print("No pages scraped – exiting.")
        return

    # 2. Clean & chunk
    all_chunks: list[tuple[str, str]] = []  # (chunk_text, source_url)
    for page in pages:
        cleaned = clean_markdown(page["markdown"])
        chunks = chunk_text(cleaned, args.chunk_size)
        for c in chunks:
            all_chunks.append((c, page["source_url"]))
    print(f"Total chunks: {len(all_chunks)}")

    # 3. Generate Q&A pairs + multi-turn (async + checkpoint)
    all_qa, all_mt = await generate_all(
        all_chunks, checkpoint_path, args.qa_per_chunk, args.model,
        args.concurrency, args.multi_turn,
    )

    print(f"Generated {len(all_qa)} raw Q&A pairs, {len(all_mt)} multi-turn conversations")

    if not all_qa and not all_mt:
        print("No Q&A pairs generated – exiting.")
        return

    # 4. Quality filter (single-turn only)
    removed_quality = 0
    filter_stats = {}
    if all_qa:
        all_qa, filter_stats = quality_filter(all_qa)
        removed_quality = sum(filter_stats.values())
        if removed_quality:
            print(f"  Quality filter removed {removed_quality} pairs: {filter_stats}")

    # 5. Deduplicate (single-turn only)
    removed_dups = 0
    if all_qa:
        all_qa, removed_dups = deduplicate(all_qa, threshold=args.dedup_threshold)
        if removed_dups:
            print(f"  Dedup removed {removed_dups} near-duplicate pairs (threshold={args.dedup_threshold})")

    print(f"  → {len(all_qa)} single-turn Q&A pairs after filtering")
    if all_mt:
        print(f"  → {len(all_mt)} multi-turn conversations")

    if not all_qa and not all_mt:
        print("No data survived filtering – exiting.")
        return

    # 6. Convert & save
    sharegpt_data = [to_sharegpt(qa) for qa in all_qa]
    alpaca_data = [to_alpaca(qa) for qa in all_qa]

    save_jsonl(sharegpt_data, output_dir / "dataset_sharegpt.jsonl")
    save_jsonl(alpaca_data, output_dir / "dataset_alpaca.jsonl")

    if all_mt:
        mt_sharegpt = [to_sharegpt_multi_turn(c) for c in all_mt]
        save_jsonl(mt_sharegpt, output_dir / "dataset_sharegpt_multi_turn.jsonl")

    # 7. Stats
    stats = {
        "url": args.url,
        "pages_scraped": len(pages),
        "total_chunks": len(all_chunks),
        "raw_qa_pairs": len(all_qa) + removed_quality + removed_dups,
        "removed_quality": removed_quality,
        "removed_quality_breakdown": filter_stats,
        "removed_duplicates": removed_dups,
        "final_qa_pairs": len(all_qa),
        "multi_turn_conversations": len(all_mt),
        "multi_turn_enabled": args.multi_turn,
        "model": args.model,
        "qa_per_chunk": args.qa_per_chunk,
        "chunk_size": args.chunk_size,
        "concurrency": args.concurrency,
        "dedup_threshold": args.dedup_threshold,
    }
    stats_path = output_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    print(f"  → Stats saved to {stats_path}")

    # 8. Clean up checkpoint on success
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("  → Checkpoint cleaned up (run completed successfully)")

    print("\nDone! Summary:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crawl a website and generate Q&A datasets for Unsloth fine-tuning."
    )
    parser.add_argument("url", help="Starting URL to crawl")
    parser.add_argument("--max-pages", type=int, default=250, help="Max pages to crawl (default: 250)")
    parser.add_argument("--output-dir", default="./output", help="Output directory (default: ./output)")
    parser.add_argument("--chunk-size", type=int, default=3000, help="Max chars per chunk (default: 3000)")
    parser.add_argument("--qa-per-chunk", type=int, default=2, help="Q&A pairs per chunk (default: 2)")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model (default: gpt-4o-mini)")
    parser.add_argument("--concurrency", type=int, default=10, help="Max concurrent OpenAI requests (default: 10)")
    parser.add_argument("--dedup-threshold", type=int, default=85, help="Fuzzy dedup similarity threshold 0-100 (default: 85)")
    parser.add_argument("--multi-turn", action="store_true", help="Also generate multi-turn conversations for ShareGPT")
    args = parser.parse_args()
    asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    main()
