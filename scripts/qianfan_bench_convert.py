#!/usr/bin/env python3
"""Convert PDFs to markdown using Qianfan OCR via a local vllm server.

This script is used by run_qianfan_benchmark.sh to process benchmark PDFs.
It converts each PDF's first page to a PNG image, sends it to a local vllm
server running baidu/Qianfan-OCR, and saves the markdown output.

The document parsing prompt and API call pattern follow the reference at:
https://github.com/baidubce/skills/blob/develop/skills/qianfanocr-document-intelligence/scripts/qianfan_ocr_cli.py
https://github.com/baidubce/skills/blob/develop/skills/qianfanocr-document-intelligence/scripts/run_document_parsing.py
"""

import base64
import glob
import json
import mimetypes
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "qianfan-ocr"

# Document parsing prompt from the reference implementation
DOCUMENT_PARSING_PROMPT = """You are an AI assistant specialized in converting document images (one or multiple pages extracted from a PDF) into Markdown with high fidelity.

Your task is to accurately convert all visible content from the images into Markdown, strictly following the rules below. Do not add explanations, comments, or inferred content.

1. Pages:
- The input may contain one or multiple page images.
- Preserve the exact page order as provided.
- If there are multiple pages, separate pages using the marker:
  --- Page N ---
  (N starts from 1)
- If there is only one page, do NOT output any page separator.

2. Text Recognition:
- Accurately convert all visible text.
- No guessing, inference, paraphrasing, or correction.
- Preserve the original document structure, including headings, paragraphs, lists, captions, and footnotes.
- Completely REMOVE all header and footer text. Do not output page numbers, running titles, or repeated marginal content.

3. Reading Order:
- Follow a top-to-bottom, left-to-right reading order.
- For multi-column layouts, fully read the left column before the right column.
- Do not reorder content for semantic or logical clarity.

4. Mathematical Formulas:
- Convert all mathematical expressions to LaTeX.
- Inline formulas must use $...$.
- Display (block) formulas must use:
  $$
  ...
  $$
- Preserve symbols, spacing, and structure exactly.
- Do not invent, simplify, normalize, or correct formulas.

5. Tables:
- Convert all tables to HTML format.
- Wrap each table with <table> and </table>.
- Preserve row and column order, merged cells (rowspan, colspan), and empty cells.
- Do not restructure or reinterpret tables.

6. Images:
- Do NOT describe image content.
- Preserve images using the exact format:
  ![label](<box>[[x1, y1, x2, y2]]</box>)
- Allowed labels: image, chart, seal.
- Completely REMOVE all header_image and footer_image elements.
- Do not introduce new labels.
- Do not remove or merge remaining image elements.

7. Unreadable or Missing Content:
- If text, symbols, or table cells are unreadable, preserve their position and leave the content empty.
- Do not guess or fill in missing information.

8. Output Requirements:
- Output Markdown only.
- Preserve original layout, spacing, and structure as closely as possible.
- Ensure clear separation between elements using line breaks.
- Do not include any explanations, metadata, or comments."""


def pdf_to_png(pdf_path, output_dir, dpi=200):
    """Convert first page of PDF to PNG using pdftoppm."""
    os.makedirs(output_dir, exist_ok=True)
    stem = Path(pdf_path).stem
    output_prefix = os.path.join(output_dir, stem)
    subprocess.run(
        ["pdftoppm", "-png", "-r", str(dpi), "-f", "1", "-l", "1", pdf_path, output_prefix],
        check=True,
        capture_output=True,
    )
    # pdftoppm outputs {prefix}-{page}.png
    candidates = sorted(glob.glob(f"{output_prefix}*.png"))
    if not candidates:
        raise RuntimeError(f"pdftoppm produced no output for {pdf_path}")
    return candidates[0]


def image_to_data_url(image_path):
    """Convert local image to base64 data URL."""
    with open(image_path, "rb") as f:
        raw = f.read()
    mime_type, _ = mimetypes.guess_type(image_path)
    mime = mime_type or "application/octet-stream"
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def strip_thinking(text):
    """Strip <think>...</think> blocks from response."""
    m = re.match(r"^\s*<think>(.*?)</think>\s*", text, re.DOTALL)
    if not m:
        return text
    return text[m.end():].strip()


def call_vllm(image_path, max_tokens=16384):
    """Send image to local vllm server and get markdown response."""
    data_url = image_to_data_url(image_path)

    body = {
        "model": MODEL_NAME,
        "max_tokens": max_tokens,
        "stream": False,
        "skip_special_tokens": False,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": DOCUMENT_PARSING_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            }
        ],
    }

    request = Request(
        VLLM_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=300) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    content = payload["choices"][0]["message"]["content"]
    return strip_thinking(content)


def main():
    pdf_root = sys.argv[1]
    target_root = sys.argv[2]

    # Find all PDFs
    pdf_files = sorted(glob.glob(os.path.join(pdf_root, "**", "*.pdf"), recursive=True))
    print(f"Found {len(pdf_files)} PDFs to process")

    with tempfile.TemporaryDirectory(prefix="qianfan_bench_") as tmpdir:
        for i, pdf_path in enumerate(pdf_files):
            rel_path = os.path.relpath(pdf_path, pdf_root)
            parts = rel_path.split(os.sep)
            if len(parts) < 2:
                print(f"Warning: Unexpected PDF path layout for {pdf_path}, skipping")
                continue

            section = parts[0]
            pdf_name = Path(pdf_path).stem

            print(f"  [{i+1}/{len(pdf_files)}] Processing {rel_path}")

            try:
                # Convert PDF to image
                png_path = pdf_to_png(pdf_path, os.path.join(tmpdir, section))

                # Call vllm API
                markdown = call_vllm(png_path)

                # Save output
                target_dir = os.path.join(target_root, section)
                os.makedirs(target_dir, exist_ok=True)
                target_path = os.path.join(target_dir, f"{pdf_name}_pg1_repeat1.md")
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(markdown)
                print(f"    -> {target_path}")
            except Exception as e:
                print(f"    Error processing {pdf_path}: {e}")
                continue

    print("Done processing all PDFs")


if __name__ == "__main__":
    main()
