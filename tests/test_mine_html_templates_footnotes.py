import random
import unittest
from unittest.mock import MagicMock, patch

from bs4 import BeautifulSoup

from olmocr.bench.synth.mine_html_templates import (
    PreserveTablesConverter,
    extract_html_metadata,
    generate_tests_from_html,
    html_to_markdown_with_frontmatter,
)
from olmocr.bench.tests import TestType


class TestFootnoteTestGeneration(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.random_gen = random.Random(42)
        self.pdf_id = "test_pdf"
        self.page_num = 1

    def test_footnote_test_generation(self):
        """Test that FootnoteTest instances are generated correctly from HTML with footnotes"""
        html_content = """
        <html>
        <body>
            <h1>Document with Footnotes</h1>
            <p>This is some text with a footnote<sup>1</sup> in the paragraph.</p>
            <p>Here is another sentence with a different footnote<sup>2</sup> marker.</p>
            <p>And yet another with footnote<sup>3</sup> reference.</p>

            <hr>
            <h3>Footnotes</h3>
            <p><sup>1</sup> This is the first footnote explaining something important about the text above.</p>
            <p><sup>2</sup> This is the second footnote with different content that provides additional information.</p>
            <p><sup>3</sup> Third footnote with more details about the referenced content in the main text.</p>
        </body>
        </html>
        """

        tests = generate_tests_from_html(html_content, self.pdf_id, self.page_num, self.random_gen, False)

        # Filter footnote tests
        footnote_tests = [t for t in tests if t.get("type") == "footnote"]

        # Should have generated footnote tests
        self.assertTrue(len(footnote_tests) == 3, "Should generate at least one footnote test")

        # Check that we have tests for footnotes 1, 2, and 3
        footnote_numbers = [test.get("marker") for test in footnote_tests]
        self.assertIn("1", footnote_numbers, "Should have footnote test for marker 1")
        self.assertIn("2", footnote_numbers, "Should have footnote test for marker 2")
        self.assertIn("3", footnote_numbers, "Should have footnote test for marker 3")



if __name__ == "__main__":
    unittest.main()