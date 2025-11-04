import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from olmocr.bench.synth.mine_html_templates import generate_tests_from_html


class TestFootnoteTestGeneration(unittest.TestCase):
    def setUp(self):
        self.random_gen = random.Random(42)
        self.pdf_id = "test_pdf"
        self.page_num = 1
        self.uuid_counter = 0
        uuid_patch = patch(
            "olmocr.bench.synth.mine_html_templates.uuid.uuid4",
            side_effect=self._fake_uuid,
        )
        self.addCleanup(uuid_patch.stop)
        uuid_patch.start()

    def _fake_uuid(self):
        current = f"{self.uuid_counter:032x}"
        self.uuid_counter += 1
        return SimpleNamespace(hex=current)

    @staticmethod
    def _hashable_tests(tests):
        return {tuple(sorted(test.items())) for test in tests}

    def _generate_footnote_tests(self, html_content):
        tests = generate_tests_from_html(
            html_content,
            self.pdf_id,
            self.page_num,
            self.random_gen,
            False,
        )
        footnote_tests = []
        for test in tests:
            if test.get("type") == "footnote":
                sanitized = {k: v for k, v in test.items() if k != "id"}
                footnote_tests.append(sanitized)
        return self._hashable_tests(footnote_tests)

    def test_single_marker_generates_marker_only(self):
        html_content = """
        <html><body>
            <p>Alpha with reference<sup>1</sup> and more text.</p>
        </body></html>
        """

        footnote_tests = self._generate_footnote_tests(html_content)

        self.assertSetEqual(
            footnote_tests,
            self._hashable_tests([
                {
                    "pdf": "test_pdf_page1.pdf",
                    "page": 1,
                    "type": "footnote",
                    "marker": "1",
                    "max_diffs": 0,
                    "marker_after": "reference",
                }
            ]),
        )

    def test_marker_with_definition_includes_text(self):
        html_content = """
        <html><body>
            <p>Alpha with reference<sup>1</sup> and more text.</p>
            <hr>
            <p><sup>1</sup>This is the footnote text that elaborates on the reference in detail.</p>
        </body></html>
        """

        footnote_tests = self._generate_footnote_tests(html_content)

        self.assertSetEqual(
            footnote_tests,
            self._hashable_tests([
                {
                    "pdf": "test_pdf_page1.pdf",
                    "page": 1,
                    "type": "footnote",
                    "marker": "1",
                    "max_diffs": 0,
                    "marker_after": "reference",
                    "text": "This is the footnote text that elaborates on the reference in detail.",
                }
            ]),
        )

    def test_multiple_markers_mixed_occurrences(self):
        html_content = """
        <html><body>
            <p>First paragraph ends with marker<sup>1</sup> in the sentence.</p>
            <p>Second paragraph carries marker<sup>2</sup> for another note.</p>
            <p>Here is a lone marker<sup>3</sup> with no definition below.</p>
            <hr>
            <p><sup>1</sup>Definition for footnote one with ample descriptive content to qualify.</p>
            <p><sup>2</sup>Definition for footnote two including enough characters to be captured correctly.</p>
        </body></html>
        """

        footnote_tests = self._generate_footnote_tests(html_content)

        self.assertSetEqual(
            footnote_tests,
            self._hashable_tests([
                {
                    "pdf": "test_pdf_page1.pdf",
                    "page": 1,
                    "type": "footnote",
                    "marker": "1",
                    "max_diffs": 0,
                    "marker_after": "marker",
                    "text": "Definition for footnote one with ample descriptive content to qualify.",
                },
                {
                    "pdf": "test_pdf_page1.pdf",
                    "page": 1,
                    "type": "footnote",
                    "marker": "2",
                    "max_diffs": 0,
                    "marker_after": "marker",
                    "text": "Definition for footnote two including enough characters to be captured correctly.",
                },
                {
                    "pdf": "test_pdf_page1.pdf",
                    "page": 1,
                    "type": "footnote",
                    "marker": "3",
                    "max_diffs": 0,
                    "marker_after": "marker",
                },
            ]),
        )



if __name__ == "__main__":
    unittest.main()
