"""Detection of text content that is cut off or hidden in rendered HTML.

Detects two types of hidden content:
1. Overflow clipping: text inside ancestors with overflow:hidden gets clipped.
2. Occlusion: opaque elements positioned on top of text, hiding it.

Watermarks and semi-transparent overlays (alpha <= 0.5) are ignored.
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
    horizontal_visible_ratio: float = 1.0  # fraction of width visible after clipping
    is_occluded: bool = False  # True if covered by an opaque non-ancestor element
    bounding_rect: dict = field(default_factory=dict)
    clipping_ancestor_tag: Optional[str] = None


@dataclass
class RenderResult:
    """Result of rendering HTML to PDF, including cutoff detection info."""

    success: bool
    scale_used: Optional[float] = None
    cutoff_elements: List[CutoffElement] = field(default_factory=list)
    has_cutoff: bool = False


_CUTOFF_JS = """
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

                    // Skip viewport clipping — PDF scaling handles viewport overflow.
                    // We only care about overflow:hidden container clipping.

                    // Calculate areas using content dimensions (including overflow)
                    const contentWidth = rect.width + overflowX;
                    const contentHeight = rect.height + overflowY;
                    const originalArea = contentWidth * contentHeight;
                    const visibleWidth = Math.max(0, visRight - visLeft);
                    const visibleHeight = Math.max(0, visBottom - visTop);
                    const visibleArea = visibleWidth * visibleHeight;

                    const visibleRatio = originalArea > 0 ? visibleArea / originalArea : 0;
                    const horizontalVisibleRatio = contentWidth > 0 ? visibleWidth / contentWidth : 1;

                    if (visibleRatio < threshold) {
                        results.push({
                            tag: el.tagName,
                            text: directText.substring(0, 500),
                            visibleRatio: visibleRatio,
                            horizontalVisibleRatio: horizontalVisibleRatio,
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
"""


_OCCLUSION_JS = """
            () => {
                const results = [];

                function getLeafTextElements(root) {
                    const leaves = [];
                    function walk(el) {
                        const style = getComputedStyle(el);
                        if (style.display === 'none' || style.visibility === 'hidden') return;
                        let hasDirectText = false;
                        for (const child of el.childNodes) {
                            if (child.nodeType === Node.TEXT_NODE && child.textContent.trim()) {
                                hasDirectText = true;
                                break;
                            }
                        }
                        if (hasDirectText) leaves.push(el);
                        for (const child of el.children) walk(child);
                    }
                    walk(root);
                    return leaves;
                }

                function isRelated(a, b) {
                    let cur = a.parentElement;
                    while (cur) { if (cur === b) return true; cur = cur.parentElement; }
                    cur = b.parentElement;
                    while (cur) { if (cur === a) return true; cur = cur.parentElement; }
                    return false;
                }

                function parseAlpha(bgColor) {
                    if (!bgColor || bgColor === 'rgba(0, 0, 0, 0)' || bgColor === 'transparent') return 0;
                    const m = bgColor.match(/rgba\\((\\d+),\\s*(\\d+),\\s*(\\d+),\\s*([\\d.]+)\\)/);
                    if (m) return parseFloat(m[4]);
                    if (bgColor.startsWith('rgb(')) return 1;
                    return 0;
                }

                const elements = getLeafTextElements(document.body);
                for (const el of elements) {
                    const rect = el.getBoundingClientRect();
                    if (rect.width <= 0 || rect.height <= 0) continue;

                    let directText = '';
                    for (const child of el.childNodes) {
                        if (child.nodeType === Node.TEXT_NODE) directText += child.textContent;
                    }
                    directText = directText.trim();
                    if (!directText || directText.length < 3) continue;

                    // Sample 5 points across the element
                    const inset = 2;
                    const pts = [
                        [rect.left + rect.width / 2, rect.top + rect.height / 2],
                        [rect.left + inset, rect.top + inset],
                        [rect.right - inset, rect.top + inset],
                        [rect.left + inset, rect.bottom - inset],
                        [rect.right - inset, rect.bottom - inset],
                    ];

                    let occludedCount = 0;
                    let blockerTag = null;
                    for (const [px, py] of pts) {
                        if (px < 0 || py < 0) continue;
                        const topEl = document.elementFromPoint(px, py);
                        if (!topEl) continue;
                        if (topEl === el || isRelated(topEl, el)) continue;

                        const alpha = parseAlpha(getComputedStyle(topEl).backgroundColor);
                        if (alpha > 0.5) {
                            occludedCount++;
                            if (!blockerTag) blockerTag = topEl.tagName;
                        }
                    }

                    // Flag if majority of sample points (3+/5) are occluded
                    if (occludedCount >= 3) {
                        results.push({
                            tag: el.tagName,
                            text: directText.substring(0, 500),
                            occludedPoints: occludedCount,
                            totalPoints: 5,
                            blockerTag: blockerTag,
                            boundingRect: {
                                left: rect.left,
                                top: rect.top,
                                right: rect.right,
                                bottom: rect.bottom,
                                width: rect.width,
                                height: rect.height
                            }
                        });
                    }
                }

                return results;
            }
"""


def _parse_cutoff_data(cutoff_data: list) -> List[CutoffElement]:
    return [
        CutoffElement(
            tag=item["tag"],
            text=item["text"],
            visible_ratio=item["visibleRatio"],
            horizontal_visible_ratio=item.get("horizontalVisibleRatio", 1.0),
            bounding_rect=item["boundingRect"],
            clipping_ancestor_tag=item.get("clippingAncestorTag"),
        )
        for item in cutoff_data
    ]


def _parse_occlusion_data(occlusion_data: list) -> List[CutoffElement]:
    return [
        CutoffElement(
            tag=item["tag"],
            text=item["text"],
            visible_ratio=1.0 - item["occludedPoints"] / item["totalPoints"],
            horizontal_visible_ratio=1.0,
            is_occluded=True,
            bounding_rect=item["boundingRect"],
            clipping_ancestor_tag=item.get("blockerTag"),
        )
        for item in occlusion_data
    ]


async def _detect_cutoff_on_page(
    page, visibility_threshold: float
) -> List[CutoffElement]:
    """Run overflow-clipping and occlusion detection on an already-loaded Playwright page."""
    cutoff_data = await page.evaluate(_CUTOFF_JS, visibility_threshold)
    occlusion_data = await page.evaluate(_OCCLUSION_JS)
    return _parse_cutoff_data(cutoff_data) + _parse_occlusion_data(occlusion_data)


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
        results = await _detect_cutoff_on_page(page, visibility_threshold)
        await browser.close()

    return results


def has_significant_cutoff(
    cutoff_elements: List[CutoffElement],
    min_text_length: int = 3,
    max_visible_ratio: float = 0.5,
) -> bool:
    """
    Determine if the detected cutoff elements represent significant content loss.

    Flags two types of issues:
    1. Horizontal overflow clipping — text width cut by overflow:hidden containers.
    2. Occlusion — text covered by opaque non-ancestor elements.

    Vertical-only clipping is ignored (PDF scaling handles it).
    Watermarks / transparent overlays are ignored by the detection JS.

    Args:
        cutoff_elements: List of CutoffElement from detect_cutoff_text.
        min_text_length: Minimum text length to consider an element significant.
        max_visible_ratio: Elements with horizontal_visible_ratio at or below this
            are considered significant cutoff (default 0.5 means >50% of the
            element's width is hidden).

    Returns:
        True if there is significant text cutoff or occlusion that warrants
        skipping or re-rendering.
    """
    for el in cutoff_elements:
        if len(el.text.strip()) < min_text_length:
            continue
        # Horizontal overflow clipping
        if el.horizontal_visible_ratio <= max_visible_ratio:
            return True
        # Occlusion by opaque element
        if el.is_occluded:
            return True
    return False


def extract_viewport_from_html(html_content: str) -> tuple:
    """Extract viewport dimensions from the HTML meta viewport tag.

    Parses ``<meta name="viewport" content="width=N, ...">``. Falls back to
    1024 wide if not found. Height always defaults to 100000 (large value
    avoids false viewport-clipping on pages that flow naturally).
    """
    width = 1024
    height = 100000
    meta_match = re.search(
        r'<meta\s[^>]*name=["\']viewport["\'][^>]*content=["\']([^"\']*)["\']'
        r'|<meta\s[^>]*content=["\']([^"\']*)["\'][^>]*name=["\']viewport["\']',
        html_content,
        re.IGNORECASE,
    )
    if meta_match:
        content = meta_match.group(1) or meta_match.group(2)
        w = re.search(r"width\s*=\s*(\d+)", content)
        if w:
            width = int(w.group(1))
        h = re.search(r"height\s*=\s*(\d+)", content)
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
    For cutoff files, a PNG screenshot is saved alongside the symlink.
    """
    os.makedirs(cutoff_dir, exist_ok=True)
    os.makedirs(no_cutoff_dir, exist_ok=True)

    n_cutoff = 0
    n_ok = 0

    async with async_playwright() as p:
        browser = await p.chromium.launch()

        for path in html_paths:
            filename = os.path.basename(path)
            try:
                with open(path, "r") as f:
                    html_content = f.read()

                vw, vh = extract_viewport_from_html(html_content)
                page = await browser.new_page(
                    viewport={"width": vw, "height": vh}
                )
                await page.set_content(html_content, wait_until="load")

                elements = await _detect_cutoff_on_page(page, visibility_threshold)
                is_cutoff = has_significant_cutoff(
                    elements, min_text_length, max_visible_ratio
                )

                if is_cutoff:
                    dest = os.path.join(cutoff_dir, filename)
                    n_cutoff += 1

                    # Save a screenshot next to the symlink
                    png_name = os.path.splitext(filename)[0] + ".png"
                    png_path = os.path.join(cutoff_dir, png_name)
                    await page.screenshot(path=png_path, full_page=False)
                else:
                    dest = os.path.join(no_cutoff_dir, filename)
                    n_ok += 1

                await page.close()

                # Symlink to the original file
                if os.path.exists(dest) or os.path.islink(dest):
                    os.remove(dest)
                os.symlink(os.path.abspath(path), dest)

                status = "CUTOFF" if is_cutoff else "ok"
                detail = ""
                if is_cutoff:
                    # Show the worst occluded element if any, otherwise worst clipped
                    occluded = [e for e in elements if e.is_occluded and len(e.text.strip()) >= min_text_length]
                    clipped = [e for e in elements if not e.is_occluded and e.horizontal_visible_ratio <= max_visible_ratio and len(e.text.strip()) >= min_text_length]
                    if occluded:
                        worst = occluded[0]
                        detail = f"  occluded: {worst.tag} by={worst.clipping_ancestor_tag} \"{worst.text[:60]}\""
                    elif clipped:
                        worst = min(clipped, key=lambda e: e.horizontal_visible_ratio)
                        detail = f"  clipped: {worst.tag} hvis={worst.horizontal_visible_ratio:.2f} \"{worst.text[:60]}\""
                    else:
                        detail = ""
                print(f"[{status:6s}] {filename}{detail}")

            except Exception as e:
                print(f"[ERROR ] {filename}: {e}")

        await browser.close()

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
