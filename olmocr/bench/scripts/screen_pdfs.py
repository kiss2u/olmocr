"""
Screen synthetic benchmark PDFs for sensitive/private content using OpenAI vision API.

Flags documents where:
  - #1 (resume/CV) is true, OR
  - #2 (sensitive PII) is true AND #3 (public consumption) is false AND #4 (academic) is false
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from olmocr.data.renderpdf import render_pdf_to_base64png

BENCH_DATA = Path(__file__).parent / "bench_data"
PDFS_DIR = BENCH_DATA / "pdfs"

SCREENING_PROMPT = """\
Look at this document image and answer each question with ONLY "yes" or "no".

1. Is this a resume or CV?
2. Does this document contain sensitive personally identifiable information (PII) such as social security numbers, personal addresses, phone numbers, dates of birth, or similar?
3. Is this document meant for public consumption or dissemination?
4. Is this document academic in nature (e.g. journal article, textbook, thesis, lecture notes)?

Respond as JSON: {"is_resume": bool, "has_sensitive_pii": bool, "is_public": bool, "is_academic": bool}
"""

MODEL = "gpt-5.4"
MAX_CONCURRENT = 5


def should_flag(result: dict) -> bool:
    """Return True if the document should be flagged for review."""
    if result.get("is_resume"):
        return True
    if result.get("has_sensitive_pii") and not result.get("is_public") and not result.get("is_academic"):
        return True
    return False


async def screen_pdf(client: AsyncOpenAI, pdf_path: str, model: str, semaphore: asyncio.Semaphore) -> dict:
    """Render PDF and send screenshot to the API, returning the parsed screening result."""
    # Render using the same pdftoppm approach as the olmocr pipeline
    image_base64 = await asyncio.to_thread(render_pdf_to_base64png, pdf_path, 1, 2048)

    async with semaphore:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_base64}", "detail": "low"},
                        },
                        {"type": "text", "text": SCREENING_PROMPT},
                    ],
                }
            ],
            temperature=0,
            max_completion_tokens=200,
        )

    text = response.choices[0].message.content.strip()
    # Extract JSON from the response (handle markdown fences)
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    parsed = json.loads(text)
    # Normalize "yes"/"no" strings to booleans
    for key in ("is_resume", "has_sensitive_pii", "is_public", "is_academic"):
        if isinstance(parsed.get(key), str):
            parsed[key] = parsed[key].lower().strip() == "yes"
    return parsed


async def process_one(client: AsyncOpenAI, category: str, pdf_path: Path, model: str, semaphore: asyncio.Semaphore) -> dict:
    """Process a single PDF and return the result entry."""
    rel = f"{category}/{pdf_path.name}"
    try:
        result = await screen_pdf(client, str(pdf_path), model, semaphore)
        flag = should_flag(result)
        return {"pdf": rel, "flag": flag, **result}
    except Exception as e:
        return {"pdf": rel, "flag": None, "error": str(e)}


async def main():
    parser = argparse.ArgumentParser(description="Screen synthetic benchmark PDFs for sensitive content")
    parser.add_argument("--model", default=MODEL, help=f"OpenAI model to use (default: {MODEL})")
    parser.add_argument("--output", default="flagged_pdfs.jsonl", help="Output file for flagged documents")
    parser.add_argument("--all-results", default=None, help="Optional file to write ALL results (not just flagged)")
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT, help=f"Max parallel API requests (default: {MAX_CONCURRENT})")
    parser.add_argument("--dry-run", action="store_true", help="List PDFs that would be screened without calling API")
    args = parser.parse_args()

    # Collect all synthetic_ folders
    synthetic_dirs = sorted(p for p in PDFS_DIR.iterdir() if p.is_dir() and p.name.startswith("synthetic_"))

    # Build list of (category, pdf_path) pairs
    pdf_files = []
    for d in synthetic_dirs:
        for f in sorted(d.iterdir()):
            if f.suffix.lower() == ".pdf":
                pdf_files.append((d.name, f))

    print(f"Found {len(pdf_files)} PDFs across {len(synthetic_dirs)} synthetic categories")

    if args.dry_run:
        for cat, p in pdf_files:
            print(f"  {cat}/{p.name}")
        return

    client = AsyncOpenAI()  # uses OPENAI_API_KEY env var
    semaphore = asyncio.Semaphore(args.max_concurrent)

    # Launch all tasks, bounded by semaphore
    tasks = [
        process_one(client, category, pdf_path, args.model, semaphore)
        for category, pdf_path in pdf_files
    ]

    all_results = []
    flagged = []
    flagged_count = 0
    error_count = 0

    pbar = tqdm(total=len(tasks), desc="Screening PDFs", unit="pdf")
    for coro in asyncio.as_completed(tasks):
        entry = await coro
        all_results.append(entry)

        if entry.get("error"):
            error_count += 1
            pbar.set_postfix(flagged=flagged_count, errors=error_count)
        elif entry["flag"]:
            flagged.append(entry)
            flagged_count += 1
            pbar.set_postfix(flagged=flagged_count, errors=error_count)

        pbar.update(1)
    pbar.close()

    # Write flagged results
    with open(args.output, "w") as f:
        for entry in flagged:
            f.write(json.dumps(entry) + "\n")
    print(f"\nWrote {len(flagged)} flagged documents to {args.output}")

    if error_count:
        print(f"  ({error_count} errors encountered)")

    # Optionally write all results
    if args.all_results:
        with open(args.all_results, "w") as f:
            for entry in all_results:
                f.write(json.dumps(entry) + "\n")
        print(f"Wrote {len(all_results)} total results to {args.all_results}")


if __name__ == "__main__":
    asyncio.run(main())
