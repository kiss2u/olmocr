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


class TestFormatTestGeneration(unittest.TestCase):
    """Test the generation of FormatTests from HTML content"""

    def setUp(self):
        """Set up test fixtures"""
        self.random_gen = random.Random(42)
        self.pdf_id = "test_pdf"
        self.page_num = 1

    def test_format_test_generation_basic(self):
        """Test basic format test generation with headings, bold, and italic"""
        html_content = """
        <html>
        <body>
            <h1>Main Title</h1>
            <h2>Subtitle</h2>
            <p>This is a paragraph with <b>bold text</b> and <i>italic text</i>.</p>
            <p>Also has <strong>strong emphasis</strong> and <em>emphasized text</em>.</p>
        </body>
        </html>
        """

        tests = generate_tests_from_html(html_content, self.pdf_id, self.page_num, self.random_gen, False)

        # Filter format tests
        format_tests = [t for t in tests if t.get("type") == "format"]

        # Should generate format tests
        self.assertGreater(len(format_tests), 0)
        self.assertLessEqual(len(format_tests), 5)  # Should be limited to 5

        # Check that we have different format types
        format_types = {t["format"] for t in format_tests}
        self.assertTrue("heading" in format_types)
        self.assertTrue("bold" in format_types or "italic" in format_types)

        # Check specific text was captured
        format_texts = {t["text"] for t in format_tests}
        self.assertIn("Main Title", format_texts)
        self.assertIn("Subtitle", format_texts)

    def test_format_test_limit(self):
        """Test that format tests are limited to 5 total"""
        html_content = """
        <html>
        <body>
            <h1>Title 1</h1>
            <h2>Title 2</h2>
            <h3>Title 3</h3>
            <h4>Title 4</h4>
            <h5>Title 5</h5>
            <h6>Title 6</h6>
            <p><b>Bold 1</b> <strong>Bold 2</strong> <b>Bold 3</b></p>
            <p><i>Italic 1</i> <em>Italic 2</em> <i>Italic 3</i></p>
        </body>
        </html>
        """

        tests = generate_tests_from_html(html_content, self.pdf_id, self.page_num, self.random_gen, False)

        # Filter format tests
        format_tests = [t for t in tests if t.get("type") == "format"]

        # Should be limited to exactly 5
        self.assertEqual(len(format_tests), 5)

    def test_format_test_with_tables_headers_footers(self):
        """Test that format tests exclude content in headers, footers, and tables"""
        html_content = """
        <html>
        <body>
            <header>
                <h1>Header Title</h1>
                <b>Header Bold</b>
            </header>

            <h2>Main Content Title</h2>
            <p>Content with <strong>bold text</strong></p>

            <table>
                <tr>
                    <td><b>Table Bold</b></td>
                    <td><i>Table Italic</i></td>
                </tr>
            </table>

            <footer>
                <em>Footer Italic</em>
            </footer>
        </body>
        </html>
        """

        tests = generate_tests_from_html(html_content, self.pdf_id, self.page_num, self.random_gen, False)

        # Filter format tests
        format_tests = [t for t in tests if t.get("type") == "format"]
        format_texts = {t["text"] for t in format_tests}

        # Should only have main content formatting, not header/footer/table content
        self.assertIn("Main Content Title", format_texts)
        self.assertIn("bold text", format_texts)
        self.assertNotIn("Header Title", format_texts)
        self.assertNotIn("Header Bold", format_texts)
        self.assertNotIn("Table Bold", format_texts)
        self.assertNotIn("Table Italic", format_texts)
        self.assertNotIn("Footer Italic", format_texts)

    def test_format_test_normalization(self):
        """Test that text normalization works in format tests"""
        html_content = """
        <html>
        <body>
            <h1>Title   with   extra   spaces</h1>
            <p><b>"Fancy" 'quotes'</b></p>
            <p><i>Text—with–dashes</i></p>
        </body>
        </html>
        """

        tests = generate_tests_from_html(html_content, self.pdf_id, self.page_num, self.random_gen, False)

        # Filter format tests
        format_tests = [t for t in tests if t.get("type") == "format"]
        format_texts = {t["text"] for t in format_tests}

        # Text should be normalized
        self.assertIn("Title with extra spaces", format_texts)
        # Quotes should be normalized
        found_quotes = any("\"Fancy\" 'quotes'" in text for text in format_texts)
        self.assertTrue(found_quotes)
        # Dashes should be normalized
        found_dashes = any("Text-with-dashes" in text for text in format_texts)
        self.assertTrue(found_dashes)

    def test_format_test_no_duplicates(self):
        """Test that duplicate text doesn't create multiple format tests"""
        html_content = """
        <html>
        <body>
            <h1>Same Title</h1>
            <h2>Same Title</h2>
            <p><b>Same Title</b></p>
            <p><i>Different Title</i></p>
        </body>
        </html>
        """

        tests = generate_tests_from_html(html_content, self.pdf_id, self.page_num, self.random_gen, False)

        # Filter format tests
        format_tests = [t for t in tests if t.get("type") == "format"]
        format_texts = [t["text"] for t in format_tests]

        # "Same Title" should only appear once
        self.assertEqual(format_texts.count("Same Title"), 1)
        # "Different Title" should also be there
        self.assertIn("Different Title", format_texts)

    def test_format_test_minimum_length(self):
        """Test that very short text is not included in format tests"""
        html_content = """
        <html>
        <body>
            <h1>A</h1>
            <h2>OK</h2>
            <h3>Valid Title</h3>
            <p><b>B</b> <strong>Good Bold</strong></p>
            <p><i>C</i> <em>Nice Italic</em></p>
        </body>
        </html>
        """

        tests = generate_tests_from_html(html_content, self.pdf_id, self.page_num, self.random_gen, False)

        # Filter format tests
        format_tests = [t for t in tests if t.get("type") == "format"]
        format_texts = {t["text"] for t in format_tests}

        # Short text (less than 3 characters) should be excluded
        self.assertNotIn("A", format_texts)
        self.assertNotIn("B", format_texts)
        self.assertNotIn("C", format_texts)
        self.assertNotIn("OK", format_texts)  # Only 2 characters

        # Longer text should be included
        self.assertIn("Valid Title", format_texts)
        self.assertIn("Good Bold", format_texts)
        self.assertIn("Nice Italic", format_texts)

    def test_bold_number_order_bug(self):
        """Test that bold numbers in HTML don't create invalid order tests with max_diffs=1

        This test catches a bug where <b>6</b> in HTML gets converted to **6** in markdown,
        which then gets selected as a sentence for order tests. Since **6** has exactly 5
        characters, it passes the length check, but when compared with a longer sentence,
        max_diffs gets set to 1 (2% of the longer sentence), which is problematic.
        """
        html_content = """
        <html>
        <body>
            <p>This is a normal sentence with some content</p>
            <p><b>6</b></p>
            <p>Another sentence here that has stuff</p>
        </body>
        </html>
        """

        tests = generate_tests_from_html(html_content, self.pdf_id, self.page_num, self.random_gen, False)

        # You don't want to see any order tests in this case, because the <b> has caused the text comparison to be too small
        order_tests = [t for t in tests if t.get("type") == "order"]
        self.assertEqual(order_tests, [])

    def test_bold_number_order_bug2(self):
        """Test that bold numbers in HTML don't create invalid order tests with max_diffs=1

        This test catches a bug where <b>6</b> in HTML gets converted to **6** in markdown,
        which then gets selected as a sentence for order tests. Since **6** has exactly 5
        characters, it passes the length check, but when compared with a longer sentence,
        max_diffs gets set to 1 (2% of the longer sentence), which is problematic.
        """
        html_content = """
        <html>
        <body>
            <p>This is a normal sentence with some content</p>
            <p><b>Wow cool!</b></p>
            <p>Another sentence here that has stuff</p>
        </body>
        </html>
        """

        tests = generate_tests_from_html(html_content, self.pdf_id, self.page_num, self.random_gen, False)

        # You don't want to see any order tests in this case, because the <b> has caused the text comparison to be too small
        order_tests = [t for t in tests if t.get("type") == "order"]
        self.assertGreater(len(order_tests), 0)

    @unittest.skip("We have commented out this type of automatic test, due to concerns it would incentivize the model wrongly")
    def test_common_words_absence(self):
        """Test that 3 common words most similar to page content are added as absence tests"""
        # Create HTML content that deliberately excludes very common English words
        # but includes words like "physical", "specific", "required" which are similar to common words
        html_content = """
        <html>
        <body>
            <h1>Scientific Research Paper</h1>
            <p>Quantum mechanics describes physical properties at atomic scale.</p>
            <p>Experiments demonstrate wave-particle duality phenomenon.</p>
            <p>Mathematical equations predict behavior accurately.</p>
            <p>Results confirm theoretical predictions remarkably well.</p>
            <p>Further investigation required regarding specific anomalies.</p>
        </body>
        </html>
        """

        # Note: This content deliberately avoids some very common words like "the", "to", "and", "of", "a", "in", "is", "for", "that"
        # But includes words that might be similar (e.g., "specific" is similar to "specify", "required" to "require")
        tests = generate_tests_from_html(html_content, self.pdf_id, self.page_num, self.random_gen, False)

        # Find absence tests for common words
        absent_common_tests = [t for t in tests if t.get("type") == "absent" and "absent_common" in t.get("id", "")]

        # We should have up to 3 absence tests for common words not on the page
        self.assertGreater(len(absent_common_tests), 0, "Should have at least one common word absence test")
        self.assertLessEqual(len(absent_common_tests), 3, "Should have at most 3 common word absence tests")

        # Check that these are indeed common words (they should be in the top 1000 most common words)
        from wordfreq import top_n_list

        top_1000_words = set(top_n_list("en", 1000))

        for test in absent_common_tests:
            word = test["text"].lower()
            self.assertIn(word, top_1000_words, f"'{word}' should be a common English word from top 1000")

        # Verify that the absent words really don't appear in the content
        markdown = html_to_markdown_with_frontmatter(html_content)
        for test in absent_common_tests:
            word = test["text"].lower()
            self.assertNotIn(word, markdown.lower(), f"'{word}' should not appear in the content")


if __name__ == "__main__":
    unittest.main()
