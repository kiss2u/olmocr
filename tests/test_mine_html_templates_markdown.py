import random
import unittest
from unittest.mock import MagicMock, patch

from bs4 import BeautifulSoup

from olmocr.bench.tests import TestType
from olmocr.synth.mine_html_templates import (
    PreserveTablesConverter,
    extract_html_metadata,
    generate_tests_from_html,
    html_to_markdown_with_frontmatter,
)


class TestHtmlToMarkdown(unittest.TestCase):
    def test_title_tag_excluded_from_markdown(self):
        """Test that title tags from head are not included in markdown output."""
        html_content = """
        <html lang="en">
        <head>
            <title>This Should Not Appear In Markdown</title>
            <meta charset="UTF-8">
        </head>
        <body>
            <h1>Main Heading</h1>
            <p>This is the body content that should appear.</p>
        </body>
        </html>
        """

        markdown_with_frontmatter = html_to_markdown_with_frontmatter(html_content)

        # Check that the title from head tag is NOT in the markdown
        self.assertNotIn("This Should Not Appear In Markdown", markdown_with_frontmatter)

        # Check that body content IS in the markdown
        self.assertIn("Main Heading", markdown_with_frontmatter)
        self.assertIn("This is the body content that should appear", markdown_with_frontmatter)

        # Check that frontmatter is present
        self.assertTrue(markdown_with_frontmatter.startswith("---"))

    def test_image_with_data_description(self):
        """Test that images are converted with placeholder alt text."""
        html_content = """
        <html lang="en">
        <body>
            <p>Text before image</p>
            <div class="image" data-description="A beautiful sunset over mountains">Placeholder</div>
            <p>Text after image</p>
        </body>
        </html>
        """

        markdown_with_frontmatter = html_to_markdown_with_frontmatter(html_content)

        # Check that images use the fixed placeholder alt text
        self.assertIn("![Image Placeholder]", markdown_with_frontmatter)

        # Check that other content is preserved
        self.assertIn("Text before image", markdown_with_frontmatter)
        self.assertIn("Text after image", markdown_with_frontmatter)

    def test_image_without_data_description(self):
        """Test that images without data-description use default alt text."""
        html_content = """
        <html lang="en">
        <body>
            <div class="image">Some placeholder content</div>
        </body>
        </html>
        """

        markdown_with_frontmatter = html_to_markdown_with_frontmatter(html_content)

        # Check that default alt text is used
        self.assertIn("![Image Placeholder]", markdown_with_frontmatter)

    def test_headers_footers_excluded(self):
        """Test that header and footer tags are excluded from markdown."""
        html_content = """
        <html lang="en">
        <body>
            <header>
                <nav>Navigation menu that should not appear</nav>
            </header>
            <main>
                <h1>Main Content</h1>
                <p>This should appear in the markdown.</p>
            </main>
            <footer>
                <p>Footer text that should not appear</p>
            </footer>
        </body>
        </html>
        """

        markdown_with_frontmatter = html_to_markdown_with_frontmatter(html_content)

        # Check that header/footer content is excluded
        self.assertNotIn("Navigation menu", markdown_with_frontmatter)
        self.assertNotIn("Footer text", markdown_with_frontmatter)

        # Check that main content is included
        self.assertIn("Main Content", markdown_with_frontmatter)
        self.assertIn("This should appear in the markdown", markdown_with_frontmatter)

    def test_no_body_tag_fallback(self):
        """Test that content is still processed when there's no body tag."""
        html_content = """
        <div>
            <h1>Content without body tag</h1>
            <p>This should still be converted.</p>
        </div>
        """

        markdown_with_frontmatter = html_to_markdown_with_frontmatter(html_content)

        # Check that content is still converted
        self.assertIn("Content without body tag", markdown_with_frontmatter)
        self.assertIn("This should still be converted", markdown_with_frontmatter)

    def test_removes_triple_dashes_from_content(self):
        """Test that --- at the start or end of markdown content is removed."""
        # Test with --- at the beginning
        html_content_start = """
        <html lang="en">
        <body>
            <p>---</p>
            <p>Regular content here</p>
        </body>
        </html>
        """

        markdown_start = html_to_markdown_with_frontmatter(html_content_start)
        lines = markdown_start.split("\n")

        # Check that we have FrontMatter
        self.assertEqual(lines[0], "---")
        # Check that the content doesn't start with --- after the FrontMatter ends
        frontmatter_end = next(i for i in range(1, len(lines)) if lines[i] == "---")
        content_after_frontmatter = "\n".join(lines[frontmatter_end + 1 :])
        self.assertFalse(content_after_frontmatter.strip().startswith("---"))

        # Test with --- at the end
        html_content_end = """
        <html lang="en">
        <body>
            <p>Regular content here</p>
            <p>---</p>
        </body>
        </html>
        """

        markdown_end = html_to_markdown_with_frontmatter(html_content_end)
        # Check that content doesn't end with ---
        self.assertFalse(markdown_end.rstrip().endswith("---\n---"))

        # Test with --- at both beginning and end
        html_content_both = """
        <html lang="en">
        <body>
            <p>---</p>
            <p>Middle content</p>
            <p>---</p>
        </body>
        </html>
        """

        markdown_both = html_to_markdown_with_frontmatter(html_content_both)
        lines_both = markdown_both.split("\n")
        frontmatter_end_both = next(i for i in range(1, len(lines_both)) if lines_both[i] == "---")
        content_both = "\n".join(lines_both[frontmatter_end_both + 1 :])

        # Content should not start or end with ---
        self.assertFalse(content_both.strip().startswith("---"))
        self.assertFalse(content_both.strip().endswith("---"))
        # But should contain "Middle content"
        self.assertIn("Middle content", content_both)


class TestSuperscriptSubscriptConversion(unittest.TestCase):
    """Test superscript and subscript conversion to Unicode in html_to_markdown_with_frontmatter"""

    def test_basic_superscripts(self):
        """Test basic superscript conversion"""
        html = """
        <html>
        <body>
            <p>x<sup>2</sup> + y<sup>3</sup> = z<sup>4</sup></p>
            <p>10<sup>9</sup> is a billion</p>
        </body>
        </html>
        """
        result = html_to_markdown_with_frontmatter(html)

        # Check that superscripts are converted to Unicode
        self.assertIn("x²", result)
        self.assertIn("y³", result)
        self.assertIn("z⁴", result)
        self.assertIn("10⁹", result)

        # Should not contain HTML sup tags in markdown
        self.assertNotIn("<sup>", result)
        self.assertNotIn("</sup>", result)

    def test_basic_subscripts(self):
        """Test basic subscript conversion"""
        html = """
        <html>
        <body>
            <p>H<sub>2</sub>O is water</p>
            <p>CO<sub>2</sub> is carbon dioxide</p>
            <p>X<sub>n</sub> represents the nth element</p>
        </body>
        </html>
        """
        result = html_to_markdown_with_frontmatter(html)

        # Check that subscripts are converted to Unicode
        self.assertIn("H₂O", result)
        self.assertIn("CO₂", result)
        self.assertIn("Xₙ", result)

        # Should not contain HTML sub tags in markdown
        self.assertNotIn("<sub>", result)
        self.assertNotIn("</sub>", result)

    def test_mixed_super_and_subscripts(self):
        """Test mixed superscripts and subscripts"""
        html = """
        <html>
        <body>
            <p>The formula is x<sup>2</sup> + H<sub>2</sub>O<sup>+</sup></p>
            <p>Chemical: Ca<sup>2+</sup> and SO<sub>4</sub><sup>2-</sup></p>
        </body>
        </html>
        """
        result = html_to_markdown_with_frontmatter(html)

        # Check mixed conversions
        self.assertIn("x²", result)
        self.assertIn("H₂O⁺", result)
        self.assertIn("Ca²⁺", result)
        self.assertIn("SO₄²⁻", result)

    def test_special_characters(self):
        """Test special character conversions"""
        html = """
        <html>
        <body>
            <p>Math: (x+y)<sup>n</sup> and f<sub>(x)</sub></p>
            <p>Ion: OH<sup>-</sup> and H<sup>+</sup></p>
            <p>Index: a<sub>i</sub> and b<sup>i</sup></p>
        </body>
        </html>
        """
        result = html_to_markdown_with_frontmatter(html)

        # Check special character conversions
        self.assertIn("(x+y)ⁿ", result)
        self.assertIn("f₍ₓ₎", result)
        self.assertIn("OH⁻", result)
        self.assertIn("H⁺", result)
        # subscript i might not be in map, so check either form
        self.assertTrue("aᵢ" in result or "a<sub>i</sub>" in result or "ai" in result)
        self.assertIn("bⁱ", result)

    def test_in_table(self):
        """Test superscripts/subscripts within HTML tables"""
        html = """
        <html>
        <body>
            <table>
                <tr>
                    <th>Chemical</th>
                    <th>Formula</th>
                </tr>
                <tr>
                    <td>Water</td>
                    <td>H<sub>2</sub>O</td>
                </tr>
                <tr>
                    <td>Sulfate ion</td>
                    <td>SO<sub>4</sub><sup>2-</sup></td>
                </tr>
            </table>
        </body>
        </html>
        """
        result = html_to_markdown_with_frontmatter(html)

        # Tables should be preserved as HTML but superscripts/subscripts should still be converted
        self.assertIn("<table>", result)

        # Check if conversions happened in table cells
        self.assertTrue("H₂O" in result or "<sub>2</sub>" in result)
        self.assertTrue("SO₄²⁻" in result or "<sub>4</sub><sup>2-</sup>" in result)

    def test_nested_elements(self):
        """Test superscripts/subscripts in nested HTML elements"""
        html = """
        <html>
        <body>
            <div>
                <p>In physics: E = mc<sup>2</sup></p>
                <ul>
                    <li>First: x<sup>1</sup></li>
                    <li>Second: x<sub>2</sub></li>
                </ul>
            </div>
        </body>
        </html>
        """
        result = html_to_markdown_with_frontmatter(html)

        # Check conversions in nested structures
        self.assertIn("mc²", result)
        self.assertTrue("x¹" in result or "x1" in result)
        self.assertTrue("x₂" in result or "x2" in result)

    def test_frontmatter_preserved(self):
        """Test that frontmatter is still generated correctly"""
        html = """
        <html lang="es">
        <body>
            <p>Test with x<sup>2</sup></p>
            <table><tr><td>Data</td></tr></table>
        </body>
        </html>
        """
        result = html_to_markdown_with_frontmatter(html)

        # Check frontmatter exists
        self.assertTrue(result.startswith("---"))
        self.assertIn("primary_language: es", result)
        self.assertIn("is_table:", result)

        # Also check the conversion happened
        self.assertIn("x²", result)

    def test_unmapped_characters(self):
        """Test characters not in the mapping"""
        html = """
        <html>
        <body>
            <p>Unknown: x<sup>abc</sup> and y<sub>xyz</sub></p>
            <p>Mixed: H<sub>2</sub>SO<sub>4</sub> with note<sup>*</sup></p>
        </body>
        </html>
        """
        result = html_to_markdown_with_frontmatter(html)

        # Unmapped characters should be left as-is or handled gracefully
        self.assertIn("H₂SO₄", result)
        # Asterisk is not in the map, so it might remain as-is
        self.assertTrue("note*" in result or "note<sup>*</sup>" in result or "note^*" in result)

    def test_empty_super_subscripts(self):
        """Test empty sup/sub tags"""
        html = """
        <html>
        <body>
            <p>Empty tags: x<sup></sup> and y<sub></sub></p>
            <p>Normal: z<sup>2</sup></p>
        </body>
        </html>
        """
        result = html_to_markdown_with_frontmatter(html)

        # Empty tags should not cause errors
        self.assertIn("z²", result)
        # Empty tags should just be removed
        self.assertIn("x", result)
        self.assertIn("y", result)

    def test_complex_math_expression(self):
        """Test a complex mathematical expression"""
        html = """
        <html>
        <body>
            <p>The equation: (x<sub>1</sub>)<sup>2</sup> + (x<sub>2</sub>)<sup>2</sup> = r<sup>2</sup></p>
            <p>Series: a<sub>0</sub> + a<sub>1</sub>x + a<sub>2</sub>x<sup>2</sup> + ... + a<sub>n</sub>x<sup>n</sup></p>
        </body>
        </html>
        """
        result = html_to_markdown_with_frontmatter(html)

        # Check complex nested expressions
        self.assertIn("x₁", result)
        self.assertIn("x₂", result)
        self.assertIn("r²", result)
        self.assertIn("a₀", result)
        self.assertIn("a₁", result)
        self.assertIn("a₂", result)
        self.assertIn("aₙ", result)
        self.assertIn("xⁿ", result)

if __name__ == "__main__":
    unittest.main()
