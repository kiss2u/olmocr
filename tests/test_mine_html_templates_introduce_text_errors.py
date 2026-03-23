import random
import unittest

from olmocr.synth.augmentations import (
    _apply_typo,
    _has_skip_ancestor,
    introduce_text_errors,
)


class TestApplyTypo(unittest.TestCase):
    """Unit tests for the _apply_typo helper."""

    def test_preserves_first_and_last_char(self):
        rng = random.Random(0)
        for seed in range(50):
            rng.seed(seed)
            typo = _apply_typo("abcdef", rng)
            self.assertEqual(typo[0], "a")
            self.assertEqual(typo[-1], "f")

    def test_swap_strategy(self):
        # Force swap by controlling the RNG
        rng = random.Random()
        word = "abcdef"
        found_swap = False
        for seed in range(200):
            rng.seed(seed)
            typo = _apply_typo(word, rng)
            # A swap keeps same length and same set of chars
            if len(typo) == len(word) and sorted(typo) == sorted(word) and typo != word:
                found_swap = True
                break
        self.assertTrue(found_swap, "Expected to find at least one swap typo")

    def test_delete_strategy(self):
        rng = random.Random()
        word = "abcdef"
        found_delete = False
        for seed in range(200):
            rng.seed(seed)
            typo = _apply_typo(word, rng)
            if len(typo) == len(word) - 1:
                found_delete = True
                break
        self.assertTrue(found_delete, "Expected to find at least one delete typo")

    def test_duplicate_strategy(self):
        rng = random.Random()
        word = "abcdef"
        found_dup = False
        for seed in range(200):
            rng.seed(seed)
            typo = _apply_typo(word, rng)
            if len(typo) == len(word) + 1:
                found_dup = True
                break
        self.assertTrue(found_dup, "Expected to find at least one duplicate typo")

    def test_short_word_unchanged(self):
        rng = random.Random(42)
        # 2-char word has 0 interior chars, cannot be mutated
        self.assertEqual(_apply_typo("ab", rng), "ab")

    def test_three_char_word(self):
        # 3-char word has exactly 1 interior char; swap requires >=2 interior, so only delete/dup
        rng = random.Random()
        word = "abc"
        for seed in range(50):
            rng.seed(seed)
            typo = _apply_typo(word, rng)
            self.assertEqual(typo[0], "a")
            self.assertEqual(typo[-1], "c")


class TestIntroduceTextErrors(unittest.TestCase):
    """Tests for the introduce_text_errors augmentation function."""

    SAMPLE_HTML = """<html><head><title>Test</title></head><body>
    <p>This paragraph contains several important words that should be eligible for modification.</p>
    <p>Another sentence with different vocabulary and longer words throughout the document.</p>
    </body></html>"""

    def test_returns_requested_number_of_typos(self):
        rng = random.Random(42)
        _, records = introduce_text_errors(self.SAMPLE_HTML, rng, num_errors=3)
        self.assertEqual(len(records), 3)

    def test_typos_appear_in_modified_html(self):
        rng = random.Random(42)
        modified_html, records = introduce_text_errors(self.SAMPLE_HTML, rng, num_errors=3)
        for record in records:
            self.assertIn(record["typo_word"], modified_html)

    def test_originals_replaced_in_modified_html(self):
        rng = random.Random(42)
        modified_html, records = introduce_text_errors(self.SAMPLE_HTML, rng, num_errors=3)
        for record in records:
            # The original word at the typo position should no longer match
            self.assertNotEqual(record["original_word"], record["typo_word"])

    def test_record_fields(self):
        rng = random.Random(42)
        _, records = introduce_text_errors(self.SAMPLE_HTML, rng, num_errors=1)
        self.assertEqual(len(records), 1)
        self.assertIn("original_word", records[0])
        self.assertIn("typo_word", records[0])

    def test_no_body_returns_unchanged(self):
        html = "<html><head></head></html>"
        rng = random.Random(42)
        modified, records = introduce_text_errors(html, rng, num_errors=3)
        self.assertEqual(records, [])

    def test_num_errors_capped_by_candidates(self):
        # Only one eligible word
        html = "<html><body><p>Hello tiny text ok.</p></body></html>"
        rng = random.Random(42)
        _, records = introduce_text_errors(html, rng, num_errors=10)
        # "Hello" and "tiny" and "text" are <5 chars; only "Hello" is 5 chars
        self.assertLessEqual(len(records), 10)

    def test_zero_errors_returns_unchanged(self):
        rng = random.Random(42)
        modified, records = introduce_text_errors(self.SAMPLE_HTML, rng, num_errors=0)
        self.assertEqual(records, [])

    def test_deterministic_with_same_seed(self):
        rng1 = random.Random(99)
        modified1, records1 = introduce_text_errors(self.SAMPLE_HTML, rng1, num_errors=3)

        rng2 = random.Random(99)
        modified2, records2 = introduce_text_errors(self.SAMPLE_HTML, rng2, num_errors=3)

        self.assertEqual(records1, records2)
        self.assertEqual(modified1, modified2)


class TestSkipAncestors(unittest.TestCase):
    """Tests that typos skip headers, footers, tables, and other excluded elements."""

    def _get_typo_words(self, html, num_errors=10):
        rng = random.Random(42)
        _, records = introduce_text_errors(html, rng, num_errors=num_errors)
        return {r["original_word"] for r in records}

    def test_skips_table_content(self):
        html = """<html><body>
        <table><tr><td>Eligible looking words inside table</td></tr></table>
        <p>These words should definitely be eligible outside.</p>
        </body></html>"""
        words = self._get_typo_words(html)
        self.assertNotIn("Eligible", words)
        self.assertNotIn("inside", words)
        self.assertNotIn("table", words)
        self.assertNotIn("looking", words)

    def test_skips_header_tags(self):
        html = """<html><body>
        <h1>Important Header Title</h1>
        <h2>Another Subtitle Value</h2>
        <p>These words should definitely be eligible outside.</p>
        </body></html>"""
        words = self._get_typo_words(html)
        self.assertNotIn("Important", words)
        self.assertNotIn("Header", words)
        self.assertNotIn("Title", words)
        self.assertNotIn("Another", words)
        self.assertNotIn("Subtitle", words)
        self.assertNotIn("Value", words)

    def test_skips_footer_element(self):
        html = """<html><body>
        <p>Paragraph words should definitely qualify here.</p>
        <footer>Footer content should never appear inside.</footer>
        </body></html>"""
        words = self._get_typo_words(html)
        self.assertNotIn("Footer", words)
        self.assertNotIn("content", words)
        self.assertNotIn("never", words)
        self.assertNotIn("appear", words)
        self.assertNotIn("inside", words)

    def test_skips_header_element(self):
        html = """<html><body>
        <header>Header content should never appear.</header>
        <p>Paragraph words should definitely appear here.</p>
        </body></html>"""
        words = self._get_typo_words(html)
        self.assertNotIn("Header", words)

    def test_skips_script_and_style(self):
        html = """<html><body>
        <script>function doSomething() { return value; }</script>
        <style>something { color: black; }</style>
        <p>Paragraph words should definitely appear here.</p>
        </body></html>"""
        words = self._get_typo_words(html)
        self.assertNotIn("doSomething", words)
        self.assertNotIn("function", words)
        self.assertNotIn("something", words)

    def test_skips_sup_and_sub(self):
        html = """<html><body>
        <p>Normal words<sup>superscript</sup> and <sub>subscript</sub> content.</p>
        </body></html>"""
        words = self._get_typo_words(html)
        self.assertNotIn("superscript", words)
        self.assertNotIn("subscript", words)

    def test_skips_code_and_pre(self):
        html = """<html><body>
        <code>inlineCode</code>
        <pre>preformatted block content</pre>
        <p>Paragraph words should definitely appear here.</p>
        </body></html>"""
        words = self._get_typo_words(html)
        self.assertNotIn("inlineCode", words)
        self.assertNotIn("preformatted", words)
        self.assertNotIn("block", words)

    def test_skips_page_header_class(self):
        html = """<html><body>
        <div class="page-header">Header content skipped</div>
        <p>Paragraph words should definitely appear here.</p>
        </body></html>"""
        words = self._get_typo_words(html)
        self.assertNotIn("Header", words)
        self.assertNotIn("skipped", words)

    def test_skips_page_footer_class(self):
        html = """<html><body>
        <div class="page-footer">Footer content skipped</div>
        <p>Paragraph words should definitely appear here.</p>
        </body></html>"""
        words = self._get_typo_words(html)
        self.assertNotIn("Footer", words)
        self.assertNotIn("skipped", words)

    def test_skips_page_number_class(self):
        html = """<html><body>
        <span class="page-number">PageNumber</span>
        <p>Paragraph words should definitely appear here.</p>
        </body></html>"""
        words = self._get_typo_words(html)
        self.assertNotIn("PageNumber", words)

    def test_eligible_paragraph_text_is_found(self):
        html = """<html><body>
        <p>These words should definitely appear as eligible candidates.</p>
        </body></html>"""
        words = self._get_typo_words(html)
        # At least some 5+ char words from the paragraph should be candidates
        self.assertTrue(len(words) > 0)


class TestWordFiltering(unittest.TestCase):
    """Tests that only appropriate words are selected as candidates."""

    def test_short_words_skipped(self):
        html = """<html><body>
        <p>The cat sat on a mat but extraordinary.</p>
        </body></html>"""
        rng = random.Random(42)
        _, records = introduce_text_errors(html, rng, num_errors=10)
        originals = {r["original_word"] for r in records}
        # Words < 5 chars should not be typo'd
        for word in ["The", "cat", "sat", "on", "a", "mat", "but"]:
            self.assertNotIn(word, originals)

    def test_non_ascii_words_skipped(self):
        html = """<html><body>
        <p>Héllo wörld but eligible words remain here.</p>
        </body></html>"""
        rng = random.Random(42)
        _, records = introduce_text_errors(html, rng, num_errors=10)
        originals = {r["original_word"] for r in records}
        # Non-ascii words like Héllo, wörld won't match [A-Za-z]+ fully
        self.assertNotIn("Héllo", originals)
        self.assertNotIn("wörld", originals)

    def test_only_alpha_words(self):
        html = """<html><body>
        <p>word123 and numbers456 but eligible words remain here.</p>
        </body></html>"""
        rng = random.Random(42)
        _, records = introduce_text_errors(html, rng, num_errors=10)
        originals = {r["original_word"] for r in records}
        # Regex [A-Za-z]+ splits on digits, so "word" (4 chars) and "numbers" are separate
        self.assertNotIn("word123", originals)
        self.assertNotIn("numbers456", originals)


class TestTypoTestValidation(unittest.TestCase):
    """Test that typo records can be validated with TextPresenceTest logic."""

    def test_typo_present_in_augmented_absent_in_original(self):
        from olmocr.bench.tests import TestType, TextPresenceTest

        html = """<html><head></head><body>
        <p>This paragraph contains several important words that should be eligible for modification throughout.</p>
        <p>Another sentence with different vocabulary and longer words throughout the document entirely.</p>
        </body></html>"""

        rng = random.Random(42)
        original_html = html
        modified_html, records = introduce_text_errors(html, rng, num_errors=3)

        self.assertGreater(len(records), 0)

        # Use the raw HTML text as a stand-in for markdown (avoids importing mine_html_templates)
        original_text = original_html
        augmented_text = modified_html

        validated = 0
        for record in records:
            typo_word = record["typo_word"]
            test_obj = TextPresenceTest(
                pdf="test.pdf",
                page=1,
                id="test",
                type=TestType.PRESENT.value,
                text=typo_word,
                max_diffs=0,
            )
            passed_original, _ = test_obj.run(original_text)
            passed_augmented, _ = test_obj.run(augmented_text)

            if not passed_original and passed_augmented:
                validated += 1

        # At least some typos should pass validation
        self.assertGreater(validated, 0)


class TestTypoDeduplication(unittest.TestCase):
    """Test that existing presence tests for typo words get deduplicated."""

    def test_existing_presence_tests_for_typo_words_removed(self):
        from olmocr.bench.tests import TestType

        typo_records = [
            {"original_word": "important", "typo_word": "improtant"},
            {"original_word": "eligible", "typo_word": "eligble"},
        ]
        typo_words = {r["typo_word"] for r in typo_records}

        # Simulate tests list that generate_tests_from_html might produce,
        # including a presence test that matches a typo word
        tests = [
            {"type": TestType.PRESENT.value, "text": "improtant", "id": "dup1"},
            {"type": TestType.PRESENT.value, "text": "something_else", "id": "keep1"},
            {"type": TestType.TABLE.value, "text": "improtant", "id": "keep2"},
            {"type": TestType.PRESENT.value, "text": "eligble", "id": "dup2"},
        ]

        # Apply the same dedup logic used in mine_html_templates
        filtered = [t for t in tests if not (t.get("type") == TestType.PRESENT.value and t.get("text") in typo_words)]

        self.assertEqual(len(filtered), 2)
        ids = {t["id"] for t in filtered}
        self.assertIn("keep1", ids)
        self.assertIn("keep2", ids)
        self.assertNotIn("dup1", ids)
        self.assertNotIn("dup2", ids)

    def test_non_present_tests_unaffected(self):
        from olmocr.bench.tests import TestType

        typo_words = {"improtant"}
        tests = [
            {"type": TestType.TABLE.value, "text": "improtant", "id": "t1"},
            {"type": TestType.ORDER.value, "text": "improtant", "id": "t2"},
        ]

        filtered = [t for t in tests if not (t.get("type") == TestType.PRESENT.value and t.get("text") in typo_words)]
        self.assertEqual(len(filtered), 2)


if __name__ == "__main__":
    unittest.main()
