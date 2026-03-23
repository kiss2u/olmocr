import os
import random
import re
from typing import Dict, List, Tuple

from bs4 import BeautifulSoup, NavigableString, Tag
from PIL import Image

from olmocr.synth.claude_client import (
    DEFAULT_MODEL_NAME,
    claude_stream,
    extract_code_block,
)


async def densify_html(client, html_content):
    """Call Claude API to generate a denser version of HTML content by doubling information density."""
    import olmocr.synth.mine_html_templates as _mine

    try:
        dense_response = await claude_stream(
            client,
            model=DEFAULT_MODEL_NAME,
            max_tokens=50000,
            temperature=0.7,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": html_content},
                        {
                            "type": "text",
                            "text": "The HTML above describes a webpage meant to render into a single printed PDF page. Please output a new full synthetic webpage that increases the amount of information on this page by 2X. "
                            "Your goal is to shrink the font size and add more synthetic content so that the general idea and structure of the page is preserved, but so that it contains twice as many final tokens. "
                            "Be careful to adjust any elements (such as footers) so that they will not overlap the main body of the newly expanded document. "
                            "But remember that it still needs to render as a single static HTML page that will print out to ONE page on a printer or in PDF form. "
                            "Output the complete revised HTML in a ```html code block.",
                        },
                    ],
                }
            ],
        )

        dense_html_text = ""
        for content in dense_response.content:
            if content.type == "text":
                dense_html_text += content.text

        # Track token usage on the main module's globals
        if hasattr(dense_response, "usage"):
            _mine.total_input_tokens += dense_response.usage.input_tokens
            _mine.total_output_tokens += dense_response.usage.output_tokens

        dense_html = extract_code_block(dense_html_text)
        if not dense_html:
            print("Warning: No HTML code block found in densifying response")
            return None

        return dense_html

    except Exception as e:
        print(f"Error calling Claude API: {e}")
        return None


def apply_jpeg_compression(pdf_path, quality, temp_dir):
    """
    Apply JPEG compression to a PDF by converting to PNG, then to JPEG, then back to PDF.

    Args:
        pdf_path: Path to the input PDF file
        quality: JPEG quality level (70-95)
        temp_dir: Directory for temporary files

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import base64
        import io

        from olmocr.data.renderpdf import render_pdf_to_base64png

        # Create temp file paths
        temp_jpeg_path = os.path.join(temp_dir, "temp_page.jpg")
        temp_pdf_path = os.path.join(temp_dir, "temp_compressed.pdf")

        # Render at high resolution for better quality
        png_base64 = render_pdf_to_base64png(pdf_path, 1, 1288)

        # Decode base64 PNG data
        png_data = base64.b64decode(png_base64)
        png_buffer = io.BytesIO(png_data)

        # Open the PNG and convert to JPEG with specified quality
        with Image.open(png_buffer) as img:
            # Convert RGBA to RGB if necessary
            if img.mode in ("RGBA", "LA", "P"):
                rgb_img = Image.new("RGB", img.size, (255, 255, 255))
                # Paste using alpha channel as mask if available
                if img.mode == "RGBA" or img.mode == "LA":
                    rgb_img.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else img.split()[1])
                else:
                    rgb_img.paste(img)
                img = rgb_img

            # Save as JPEG with specified quality
            img.save(temp_jpeg_path, "JPEG", quality=quality, optimize=True)

            # Convert JPEG back to PDF
            img_for_pdf = Image.open(temp_jpeg_path)
            img_for_pdf.save(temp_pdf_path, "PDF", resolution=100.0)

        # Replace original PDF with compressed version
        os.replace(temp_pdf_path, pdf_path)

        # Clean up temp files
        if os.path.exists(temp_jpeg_path):
            os.remove(temp_jpeg_path)

        return True

    except Exception as e:
        print(f"Error applying JPEG compression: {e}")
        return False


_SKIP_ANCESTORS = frozenset(
    {
        "header",
        "footer",
        "table",
        "thead",
        "tbody",
        "tfoot",
        "tr",
        "td",
        "th",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "sup",
        "sub",
        "script",
        "style",
        "code",
        "pre",
    }
)
_SKIP_CLASSES = frozenset({"page-header", "page-footer", "page-number"})


def _has_skip_ancestor(node):
    """Return True if any ancestor of *node* should be excluded from typo injection."""
    for parent in node.parents:
        if parent.name in _SKIP_ANCESTORS:
            return True
        parent_classes = parent.get("class", []) if hasattr(parent, "get") else []
        if any(c in _SKIP_CLASSES for c in parent_classes):
            return True
    return False


def _apply_typo(word: str, rng: random.Random) -> str:
    """Apply a random typo to *word*, preserving the first and last character.

    Strategies:
      - swap:  swap two adjacent interior characters
      - delete: remove a random interior character
      - duplicate: double a random interior character
    """
    # Interior indices: 1 .. len(word)-2
    interior_len = len(word) - 2
    if interior_len < 1:
        return word  # too short to mutate safely

    strategy = rng.choice(["swap", "delete", "duplicate"])
    chars = list(word)

    if strategy == "swap" and interior_len >= 2:
        i = rng.randint(1, len(word) - 3)  # i and i+1 are both interior
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    elif strategy == "delete":
        i = rng.randint(1, len(word) - 2)
        del chars[i]
    else:  # duplicate (also fallback when swap needs >=2 interior chars)
        i = rng.randint(1, len(word) - 2)
        chars.insert(i, chars[i])

    return "".join(chars)


def introduce_text_errors(html_content: str, random_gen: random.Random, num_errors: int = 5) -> Tuple[str, List[Dict[str, str]]]:
    """Introduce intentional typos into body text of *html_content*.

    Returns (modified_html, typo_records) where each record is
    ``{"original_word": ..., "typo_word": ...}``.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    body = soup.find("body")
    if not body or not isinstance(body, Tag):
        return html_content, []

    # Collect candidate (text_node, word, start, end) tuples
    _WORD_RE = re.compile(r"[A-Za-z]+")
    candidates = []

    for text_node in body.find_all(string=True):
        if not isinstance(text_node, NavigableString):
            continue
        if _has_skip_ancestor(text_node):
            continue
        text = str(text_node)
        for m in _WORD_RE.finditer(text):
            word = m.group()
            if len(word) >= 5 and word.isascii():
                candidates.append((text_node, word, m.start(), m.end()))

    if not candidates:
        return html_content, []

    random_gen.shuffle(candidates)
    selected = candidates[:num_errors]

    # Group selected candidates by text node so we can replace right-to-left
    from collections import defaultdict

    node_edits: Dict[NavigableString, list] = defaultdict(list)
    typo_records: List[Dict[str, str]] = []

    for text_node, word, start, end in selected:
        typo = _apply_typo(word, random_gen)
        if typo == word:
            continue
        node_edits[text_node].append((start, end, typo))
        typo_records.append({"original_word": word, "typo_word": typo})

    # Apply edits right-to-left within each node to preserve positions
    for text_node, edits in node_edits.items():
        text = str(text_node)
        for start, end, typo in sorted(edits, key=lambda e: e[0], reverse=True):
            text = text[:start] + typo + text[end:]
        text_node.replace_with(NavigableString(text))

    return str(soup), typo_records
