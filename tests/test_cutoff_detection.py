import asyncio
import unittest

from olmocr.synth.cutoff_detection import (
    CutoffElement,
    detect_cutoff_text,
    has_significant_cutoff,
)


class TestDetectCutoffText(unittest.TestCase):
    """Tests for detect_cutoff_text using Playwright to detect overflow-clipped text."""

    def test_no_cutoff_simple_page(self):
        """All text fits within the viewport and no overflow:hidden clipping."""
        html = """<!DOCTYPE html>
<html><head><style>
body { margin: 0; padding: 10px; width: 400px; }
</style></head>
<body>
<h1>Hello World</h1>
<p>This is a simple paragraph that fits within the viewport.</p>
</body></html>"""
        results = asyncio.run(detect_cutoff_text(html, 400, 300))
        self.assertEqual(len(results), 0, f"Expected no cutoff, got: {results}")

    def test_cutoff_overflow_hidden_horizontal(self):
        """Text inside a narrow overflow:hidden container is clipped horizontally."""
        html = """<!DOCTYPE html>
<html><head><style>
body { margin: 0; }
.container {
    width: 100px;
    overflow: hidden;
    white-space: nowrap;
}
</style></head>
<body>
<div class="container">
    <p>This is a very long sentence that will definitely overflow the 100px wide container and get clipped</p>
</div>
</body></html>"""
        results = asyncio.run(detect_cutoff_text(html, 800, 600))
        self.assertGreater(len(results), 0, "Should detect clipped text")
        # The paragraph text should be the one flagged
        clipped_texts = [r.text for r in results]
        self.assertTrue(
            any("very long sentence" in t for t in clipped_texts),
            f"Expected to find the clipped sentence, got: {clipped_texts}",
        )
        # The clipping ancestor should be the container div
        for r in results:
            if "very long sentence" in r.text:
                self.assertLess(r.visible_ratio, 0.9)
                self.assertEqual(r.clipping_ancestor_tag, "DIV")

    def test_cutoff_overflow_hidden_vertical(self):
        """Text inside a short overflow:hidden container is clipped vertically."""
        html = """<!DOCTYPE html>
<html><head><style>
body { margin: 0; }
.container {
    width: 200px;
    height: 20px;
    overflow: hidden;
}
</style></head>
<body>
<div class="container">
    <p>Line one</p>
    <p>Line two should be hidden</p>
    <p>Line three should be hidden</p>
</div>
</body></html>"""
        results = asyncio.run(detect_cutoff_text(html, 800, 600))
        # At least the later lines should be detected as cut off
        clipped_texts = [r.text for r in results]
        self.assertTrue(
            any("hidden" in t for t in clipped_texts),
            f"Expected clipped lines, got: {clipped_texts}",
        )

    def test_cutoff_table_in_narrow_container(self):
        """A wide table inside a narrow overflow:hidden container - similar to the real issue."""
        html = """<!DOCTYPE html>
<html><head><style>
body { margin: 0; }
.column {
    width: 200px;
    overflow: hidden;
}
table { border-collapse: collapse; width: auto; }
th, td { border: 1px solid #999; padding: 4px 8px; white-space: nowrap; }
</style></head>
<body>
<div class="column">
    <table>
        <tr>
            <th>Name</th>
            <th>Column A</th>
            <th>Column B</th>
            <th>Column C - This Should Be Cutoff</th>
            <th>Column D - Also Cutoff</th>
        </tr>
        <tr>
            <td>Row 1</td>
            <td>Value A1</td>
            <td>Value B1</td>
            <td>Value C1</td>
            <td>Value D1</td>
        </tr>
    </table>
</div>
</body></html>"""
        results = asyncio.run(detect_cutoff_text(html, 800, 600))
        clipped_texts = [r.text for r in results]
        # The rightmost columns should be flagged
        self.assertTrue(
            any("Cutoff" in t for t in clipped_texts),
            f"Expected table column cutoff, got: {clipped_texts}",
        )

    def test_no_cutoff_table_fits(self):
        """A small table that fits within its container should not be flagged."""
        html = """<!DOCTYPE html>
<html><head><style>
body { margin: 0; }
.column {
    width: 600px;
    overflow: hidden;
}
table { border-collapse: collapse; }
th, td { border: 1px solid #999; padding: 4px 8px; }
</style></head>
<body>
<div class="column">
    <table>
        <tr>
            <th>Name</th>
            <th>Value</th>
        </tr>
        <tr>
            <td>Item 1</td>
            <td>100</td>
        </tr>
    </table>
</div>
</body></html>"""
        results = asyncio.run(detect_cutoff_text(html, 800, 600))
        self.assertEqual(len(results), 0, f"Expected no cutoff, got: {results}")

    def test_cutoff_viewport_clip(self):
        """Text positioned outside the viewport should be detected."""
        html = """<!DOCTYPE html>
<html><head><style>
body { margin: 0; overflow: hidden; width: 300px; height: 200px; }
.offscreen {
    position: absolute;
    left: 500px;
    top: 0;
}
</style></head>
<body>
<p>Visible text</p>
<p class="offscreen">This text is off screen to the right</p>
</body></html>"""
        results = asyncio.run(detect_cutoff_text(html, 300, 200))
        clipped_texts = [r.text for r in results]
        self.assertTrue(
            any("off screen" in t for t in clipped_texts),
            f"Expected offscreen text detected, got: {clipped_texts}",
        )

    def test_completely_hidden_element(self):
        """An element fully hidden by overflow:hidden should have visible_ratio near 0."""
        html = """<!DOCTYPE html>
<html><head><style>
body { margin: 0; }
.container {
    width: 200px;
    height: 30px;
    overflow: hidden;
}
.hidden-content {
    margin-top: 100px;
}
</style></head>
<body>
<div class="container">
    <p>Visible line</p>
    <p class="hidden-content">Completely hidden text below</p>
</div>
</body></html>"""
        results = asyncio.run(detect_cutoff_text(html, 800, 600))
        hidden_elements = [r for r in results if "Completely hidden" in r.text]
        self.assertGreater(len(hidden_elements), 0, "Should detect fully hidden element")
        for el in hidden_elements:
            self.assertAlmostEqual(el.visible_ratio, 0.0, places=1)

    def test_visibility_threshold(self):
        """Lower threshold should return fewer results."""
        html = """<!DOCTYPE html>
<html><head><style>
body { margin: 0; }
.container {
    width: 150px;
    overflow: hidden;
    white-space: nowrap;
}
</style></head>
<body>
<div class="container">
    <p>Short text that is slightly clipped at the edge</p>
</div>
</body></html>"""
        # With strict threshold (0.99), flag even slightly clipped elements
        strict_results = asyncio.run(detect_cutoff_text(html, 800, 600, visibility_threshold=0.99))
        # With lenient threshold (0.1), only flag severely clipped
        lenient_results = asyncio.run(detect_cutoff_text(html, 800, 600, visibility_threshold=0.1))
        self.assertGreaterEqual(len(strict_results), len(lenient_results))

    def test_display_none_not_flagged(self):
        """Elements with display:none should not be flagged as cutoff."""
        html = """<!DOCTYPE html>
<html><head><style>
body { margin: 0; }
.hidden { display: none; }
</style></head>
<body>
<p>Visible text</p>
<p class="hidden">This is intentionally hidden via display:none</p>
</body></html>"""
        results = asyncio.run(detect_cutoff_text(html, 800, 600))
        clipped_texts = [r.text for r in results]
        self.assertFalse(
            any("intentionally hidden" in t for t in clipped_texts),
            f"display:none elements should not be flagged, got: {clipped_texts}",
        )

    def test_multicolumn_poster_layout(self):
        """Simulates the real poster issue: a wide table in a narrow grid column."""
        html = """<!DOCTYPE html>
<html><head><style>
body { margin: 0; width: 1024px; height: 724px; overflow: hidden; font-size: 6px; }
.main-content {
    display: grid;
    grid-template-columns: 160px 170px 180px 170px 180px 150px;
    gap: 2px;
    padding: 2px;
    width: 1024px;
    overflow: hidden;
}
.col { overflow: hidden; }
.section-box { border: 1px solid #aaa; padding: 2px; overflow: hidden; }
table { border-collapse: collapse; width: 100%; font-size: 5.5px; }
th, td { border: 1px solid #999; padding: 1px 2px; text-align: center; white-space: nowrap; }
</style></head>
<body>
<div class="main-content">
    <div class="col"></div>
    <div class="col"></div>
    <div class="col"></div>
    <div class="col"></div>
    <div class="col">
        <div class="section-box">
            <table>
                <tr>
                    <th rowspan="2">Outcome</th>
                    <th colspan="2">0-12 weeks OR (95% CI)</th>
                    <th colspan="2">&gt;12 weeks OR (95% CI)</th>
                    <th colspan="2">All weeks OR (95% CI)</th>
                </tr>
                <tr>
                    <th>Unweighted</th>
                    <th>Weighted</th>
                    <th>Unweighted model</th>
                    <th>Weighted model</th>
                    <th>Unweighted model</th>
                    <th>Weighted model</th>
                </tr>
                <tr>
                    <td>Death</td>
                    <td>0.84</td>
                    <td>1.03</td>
                    <td>0.53</td>
                    <td>0.93</td>
                    <td>Hidden value 1</td>
                    <td>Hidden value 2</td>
                </tr>
            </table>
        </div>
    </div>
    <div class="col"></div>
</div>
</body></html>"""
        results = asyncio.run(detect_cutoff_text(html, 1024, 724))
        # At minimum, some of the wide table content should be flagged
        self.assertGreater(
            len(results), 0,
            "Should detect cutoff in the wide table within narrow grid column",
        )

    def test_real_poster_html(self):
        """Test with the actual poster HTML file if it exists."""
        import os
        poster_path = "/home/ubuntu/olmocr/synth_testposter1/html/synthetic/pdf_00000_page1.html"
        if not os.path.exists(poster_path):
            self.skipTest("Poster HTML file not available")

        with open(poster_path, "r") as f:
            html_content = f.read()

        results = asyncio.run(detect_cutoff_text(html_content, 1024, 724))
        # We know this poster has cutoff table columns
        self.assertGreater(len(results), 0, "Should detect cutoff in real poster HTML")

        # At least some table-related content should be flagged
        has_table_cutoff = any(
            r.tag in ("TH", "TD") and r.visible_ratio < 0.9
            for r in results
        )
        self.assertTrue(
            has_table_cutoff,
            f"Expected table cell cutoff detection. Clipped elements: "
            f"{[(r.tag, r.text[:50], r.visible_ratio) for r in results[:10]]}",
        )


class TestHasSignificantCutoff(unittest.TestCase):
    """Tests for the has_significant_cutoff helper function."""

    def test_no_elements(self):
        self.assertFalse(has_significant_cutoff([]))

    def test_short_text_not_significant(self):
        """Very short clipped text (e.g., a period or dash) is not significant."""
        elements = [
            CutoffElement(tag="SPAN", text=".", visible_ratio=0.0),
            CutoffElement(tag="SPAN", text="-", visible_ratio=0.0),
        ]
        self.assertFalse(has_significant_cutoff(elements, min_text_length=3))

    def test_long_text_significantly_cutoff(self):
        """Long text that is mostly hidden is significant."""
        elements = [
            CutoffElement(
                tag="TH",
                text=">12 weeks OR (95% CI)",
                visible_ratio=0.1,
            ),
        ]
        self.assertTrue(has_significant_cutoff(elements))

    def test_long_text_barely_cutoff_not_significant(self):
        """Long text that is mostly visible (e.g., 80%) is not significant at default threshold."""
        elements = [
            CutoffElement(
                tag="P",
                text="Some paragraph text that is mostly visible",
                visible_ratio=0.8,
            ),
        ]
        # Default max_visible_ratio is 0.5, so 0.8 should not be significant
        self.assertFalse(has_significant_cutoff(elements))

    def test_custom_thresholds(self):
        elements = [
            CutoffElement(
                tag="TD",
                text="Value C1",
                visible_ratio=0.6,
            ),
        ]
        # At default threshold (0.5), 0.6 is NOT significant
        self.assertFalse(has_significant_cutoff(elements, max_visible_ratio=0.5))
        # At stricter threshold (0.7), 0.6 IS significant
        self.assertTrue(has_significant_cutoff(elements, max_visible_ratio=0.7))

    def test_mixed_elements(self):
        """Only needs one significant element to return True."""
        elements = [
            CutoffElement(tag="P", text="Visible", visible_ratio=0.95),
            CutoffElement(tag="SPAN", text=".", visible_ratio=0.0),  # too short
            CutoffElement(tag="TH", text="All weeks OR (95% CI)", visible_ratio=0.0),
        ]
        self.assertTrue(has_significant_cutoff(elements))


if __name__ == "__main__":
    unittest.main()
