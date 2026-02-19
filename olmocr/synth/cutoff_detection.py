"""Detection of text content that is cut off or hidden due to overflow clipping in rendered HTML.

When HTML pages are rendered to PDF using Playwright, elements contained in ancestors
with overflow:hidden can be clipped, making text invisible in the final output. This
module provides utilities to detect such clipped content.
"""

import argparse
import asyncio
import glob
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

from playwright.async_api import async_playwright


@dataclass
class CutoffElement:
    """Represents an element whose text content is partially or fully cut off."""

    tag: str
    text: str
    visible_ratio: float  # 0.0 = fully hidden, 1.0 = fully visible
    bounding_rect: dict = field(default_factory=dict)
    clipping_ancestor_tag: Optional[str] = None


async def detect_cutoff_text(
    html_content: str,
    viewport_width: int,
    viewport_height: int,
    visibility_threshold: float = 0.9,
) -> List[CutoffElement]:
    """
    Detect text elements that are cut off due to overflow:hidden or viewport clipping.

    Renders the HTML in a headless browser and checks each text-containing element's
    bounding box against all ancestor elements that have overflow:hidden. If the
    element's visible area (after clipping) is less than the threshold fraction of
    its full area, it is flagged.

    Args:
        html_content: The HTML string to analyze.
        viewport_width: Width of the viewport in pixels.
        viewport_height: Height of the viewport in pixels.
        visibility_threshold: Elements with visible_ratio below this are flagged (0.0-1.0).

    Returns:
        List of CutoffElement instances for elements that are significantly cut off.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(
            viewport={"width": viewport_width, "height": viewport_height}
        )
        await page.set_content(html_content, wait_until="load")

        cutoff_data = await page.evaluate(
            """
            (threshold) => {
                const results = [];

                // Collect leaf text elements: elements that have at least one
                // direct Text node child with non-whitespace content.
                function getLeafTextElements(root) {
                    const leaves = [];
                    function walk(el) {
                        // Skip invisible elements
                        const style = getComputedStyle(el);
                        if (style.display === 'none' || style.visibility === 'hidden') {
                            return;
                        }

                        let hasDirectText = false;
                        for (const child of el.childNodes) {
                            if (child.nodeType === Node.TEXT_NODE && child.textContent.trim()) {
                                hasDirectText = true;
                                break;
                            }
                        }

                        if (hasDirectText) {
                            leaves.push(el);
                        }

                        for (const child of el.children) {
                            walk(child);
                        }
                    }
                    walk(root);
                    return leaves;
                }

                const elements = getLeafTextElements(document.body);

                for (const el of elements) {
                    const rect = el.getBoundingClientRect();
                    if (rect.width <= 0 || rect.height <= 0) continue;

                    // Only consider the direct text content of this element
                    let directText = '';
                    for (const child of el.childNodes) {
                        if (child.nodeType === Node.TEXT_NODE) {
                            directText += child.textContent;
                        }
                    }
                    directText = directText.trim();
                    if (!directText) continue;

                    // Use scrollWidth/scrollHeight to account for content that
                    // overflows the element's CSS box (e.g. white-space: nowrap
                    // text inside a fixed-width block element).
                    const overflowX = Math.max(0, el.scrollWidth - el.clientWidth);
                    const overflowY = Math.max(0, el.scrollHeight - el.clientHeight);
                    const contentRight = rect.right + overflowX;
                    const contentBottom = rect.bottom + overflowY;

                    // Compute the visible rect by intersecting with all clipping ancestors
                    let visLeft = rect.left;
                    let visTop = rect.top;
                    let visRight = contentRight;
                    let visBottom = contentBottom;

                    let clippingAncestorTag = null;
                    let ancestor = el.parentElement;

                    while (ancestor) {
                        const aStyle = getComputedStyle(ancestor);
                        const ovf = aStyle.overflow;
                        const ovfX = aStyle.overflowX;
                        const ovfY = aStyle.overflowY;

                        const clipsX = ovf === 'hidden' || ovf === 'clip' ||
                                       ovfX === 'hidden' || ovfX === 'clip';
                        const clipsY = ovf === 'hidden' || ovf === 'clip' ||
                                       ovfY === 'hidden' || ovfY === 'clip';

                        if (clipsX || clipsY) {
                            const aRect = ancestor.getBoundingClientRect();

                            const prevLeft = visLeft;
                            const prevTop = visTop;
                            const prevRight = visRight;
                            const prevBottom = visBottom;

                            if (clipsX) {
                                visLeft = Math.max(visLeft, aRect.left);
                                visRight = Math.min(visRight, aRect.right);
                            }
                            if (clipsY) {
                                visTop = Math.max(visTop, aRect.top);
                                visBottom = Math.min(visBottom, aRect.bottom);
                            }

                            // Record the first ancestor that actually clips this element
                            if (!clippingAncestorTag &&
                                (visLeft !== prevLeft || visTop !== prevTop ||
                                 visRight !== prevRight || visBottom !== prevBottom)) {
                                clippingAncestorTag = ancestor.tagName;
                            }
                        }

                        ancestor = ancestor.parentElement;
                    }

                    // Also clip to viewport
                    const prevLeft = visLeft;
                    const prevTop = visTop;
                    const prevRight = visRight;
                    const prevBottom = visBottom;

                    visLeft = Math.max(visLeft, 0);
                    visTop = Math.max(visTop, 0);
                    visRight = Math.min(visRight, window.innerWidth);
                    visBottom = Math.min(visBottom, window.innerHeight);

                    if (!clippingAncestorTag &&
                        (visLeft !== prevLeft || visTop !== prevTop ||
                         visRight !== prevRight || visBottom !== prevBottom)) {
                        clippingAncestorTag = 'VIEWPORT';
                    }

                    // Calculate areas using content dimensions (including overflow)
                    const contentWidth = rect.width + overflowX;
                    const contentHeight = rect.height + overflowY;
                    const originalArea = contentWidth * contentHeight;
                    const visibleWidth = Math.max(0, visRight - visLeft);
                    const visibleHeight = Math.max(0, visBottom - visTop);
                    const visibleArea = visibleWidth * visibleHeight;

                    const visibleRatio = originalArea > 0 ? visibleArea / originalArea : 0;

                    if (visibleRatio < threshold) {
                        results.push({
                            tag: el.tagName,
                            text: directText.substring(0, 500),
                            visibleRatio: visibleRatio,
                            boundingRect: {
                                left: rect.left,
                                top: rect.top,
                                right: rect.right,
                                bottom: rect.bottom,
                                width: rect.width,
                                height: rect.height
                            },
                            clippingAncestorTag: clippingAncestorTag
                        });
                    }
                }

                return results;
            }
        """,
            visibility_threshold,
        )

        await browser.close()

    return [
        CutoffElement(
            tag=item["tag"],
            text=item["text"],
            visible_ratio=item["visibleRatio"],
            bounding_rect=item["boundingRect"],
            clipping_ancestor_tag=item.get("clippingAncestorTag"),
        )
        for item in cutoff_data
    ]


def has_significant_cutoff(
    cutoff_elements: List[CutoffElement],
    min_text_length: int = 3,
    max_visible_ratio: float = 0.5,
) -> bool:
    """
    Determine if the detected cutoff elements represent significant content loss.

    Args:
        cutoff_elements: List of CutoffElement from detect_cutoff_text.
        min_text_length: Minimum text length to consider an element significant.
        max_visible_ratio: Elements with visible_ratio at or below this are considered
            significant cutoff (default 0.5 means >50% of the element is hidden).

    Returns:
        True if there is significant text cutoff that warrants skipping or re-rendering.
    """
    for el in cutoff_elements:
        if len(el.text.strip()) >= min_text_length and el.visible_ratio <= max_visible_ratio:
            return True
    return False


def extract_viewport_from_html(html_content: str) -> tuple:
    """Extract viewport width and height from the HTML body style.

    Looks for width/height in the body CSS rule. Falls back to 1024x768
    if not found.
    """
    # Try body { ... width: Npx; ... height: Npx; ... }
    body_match = re.search(r"body\s*\{([^}]*)\}", html_content)
    width, height = 1024, 768
    if body_match:
        body_css = body_match.group(1)
        w = re.search(r"width\s*:\s*(\d+)px", body_css)
        h = re.search(r"height\s*:\s*(\d+)px", body_css)
        if w:
            width = int(w.group(1))
        if h:
            height = int(h.group(1))
    return width, height


async def classify_html_files(
    html_paths: List[str],
    cutoff_dir: str,
    no_cutoff_dir: str,
    visibility_threshold: float = 0.9,
    min_text_length: int = 3,
    max_visible_ratio: float = 0.5,
):
    """Classify HTML files into cutoff / no-cutoff directories.

    Files are symlinked (not copied) into the output directories.
    """
    os.makedirs(cutoff_dir, exist_ok=True)
    os.makedirs(no_cutoff_dir, exist_ok=True)

    n_cutoff = 0
    n_ok = 0

    for path in html_paths:
        filename = os.path.basename(path)
        try:
            with open(path, "r") as f:
                html_content = f.read()

            vw, vh = extract_viewport_from_html(html_content)
            elements = await detect_cutoff_text(
                html_content, vw, vh, visibility_threshold
            )
            is_cutoff = has_significant_cutoff(
                elements, min_text_length, max_visible_ratio
            )

            if is_cutoff:
                dest = os.path.join(cutoff_dir, filename)
                n_cutoff += 1
            else:
                dest = os.path.join(no_cutoff_dir, filename)
                n_ok += 1

            # Symlink to the original file
            if os.path.exists(dest) or os.path.islink(dest):
                os.remove(dest)
            os.symlink(os.path.abspath(path), dest)

            status = "CUTOFF" if is_cutoff else "ok"
            detail = ""
            if is_cutoff:
                worst = min(elements, key=lambda e: e.visible_ratio)
                detail = f"  worst: {worst.tag} vis={worst.visible_ratio:.2f} \"{worst.text[:60]}\""
            print(f"[{status:6s}] {filename}{detail}")

        except Exception as e:
            print(f"[ERROR ] {filename}: {e}")

    print(f"\nDone. cutoff={n_cutoff}  ok={n_ok}  total={n_cutoff + n_ok}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sort HTML files into cutoff / no-cutoff directories based on overflow clipping detection."
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing HTML files to classify.",
    )
    parser.add_argument(
        "--cutoff-dir",
        default=None,
        help="Output directory for files with cutoff (default: <input_dir>_cutoff).",
    )
    parser.add_argument(
        "--no-cutoff-dir",
        default=None,
        help="Output directory for files without cutoff (default: <input_dir>_no_cutoff).",
    )
    parser.add_argument(
        "--visibility-threshold",
        type=float,
        default=0.9,
        help="Flag elements with visible ratio below this (default: 0.9).",
    )
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=3,
        help="Minimum text length to consider significant (default: 3).",
    )
    parser.add_argument(
        "--max-visible-ratio",
        type=float,
        default=0.5,
        help="Elements at or below this visibility are 'significant' cutoff (default: 0.5).",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.rstrip("/")
    cutoff_dir = args.cutoff_dir or f"{input_dir}_cutoff"
    no_cutoff_dir = args.no_cutoff_dir or f"{input_dir}_no_cutoff"

    html_paths = sorted(glob.glob(os.path.join(input_dir, "*.html")))
    if not html_paths:
        print(f"No .html files found in {input_dir}")
        exit(1)

    print(f"Found {len(html_paths)} HTML files in {input_dir}")
    print(f"  cutoff   -> {cutoff_dir}")
    print(f"  no_cutoff -> {no_cutoff_dir}")
    print()

    asyncio.run(
        classify_html_files(
            html_paths,
            cutoff_dir,
            no_cutoff_dir,
            visibility_threshold=args.visibility_threshold,
            min_text_length=args.min_text_length,
            max_visible_ratio=args.max_visible_ratio,
        )
    )
