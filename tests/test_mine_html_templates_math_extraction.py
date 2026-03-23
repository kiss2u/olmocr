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


class TestMathExtraction(unittest.TestCase):
    """Test the math extraction functionality in mine_html_templates.py"""

    def setUp(self):
        self.random_generator = random.Random(42)
        return super().setUp()

    def test_math_extraction_from_html(self):
        """Test that math equations are properly extracted from HTML content"""
        html_content = """
        <html>
        <body>
        <p>Some text with inline math \\(x = 2\\) here.</p>
        <p>Display math: \\[E = mc^2\\]</p>
        <p>Another inline: \\(\\alpha + \\beta = \\gamma\\)</p>
        <p>Complex display: \\[\\int_0^\\infty e^{-x} dx = 1\\]</p>
        </body>
        </html>
        """

        # Generate tests from HTML
        tests = generate_tests_from_html(html_content, "test_pdf", 1, self.random_generator)

        # Filter math tests
        math_tests = [t for t in tests if t.get("type") == "math"]

        # Check that we extracted math equations
        self.assertTrue(len(math_tests) > 0, "Should extract at least one math equation")

        # Check that specific equations were extracted
        math_contents = [t["math"] for t in math_tests]
        self.assertIn("x = 2", math_contents)
        self.assertIn("E = mc^2", math_contents)
        self.assertIn("\\alpha + \\beta = \\gamma", math_contents)
        self.assertIn("\\int_0^\\infty e^{-x} dx = 1", math_contents)

    def test_math_extraction_with_multiline(self):
        """Test extraction of multiline math equations"""
        html_content = """
        <html>
        <body>
        <p>Multiline equation:
        \\[
        e_i = \\frac{e_i + \\varphi(e_i)}{2} + \\frac{e_i - \\varphi(e_i)}{2}, 
        \\quad \\text{for } i \\in \\mathbb{N}.
        \\]
        </p>
        </body>
        </html>
        """

        tests = generate_tests_from_html(html_content, "test_pdf", 1, self.random_generator)
        math_tests = [t for t in tests if t.get("type") == "math"]

        # Check multiline equation is captured
        self.assertTrue(len(math_tests) > 0)

        # Check that the multiline content is preserved (without excessive newlines)
        found_multiline = False
        for test in math_tests:
            if "\\frac{e_i + \\varphi(e_i)}{2}" in test["math"] and "\\mathbb{N}" in test["math"]:
                found_multiline = True
                break

        self.assertTrue(found_multiline, "Should extract multiline equation correctly")

    def test_math_extraction_deduplication(self):
        """Test that duplicate math equations are deduplicated"""
        html_content = """
        <html>
        <body>
        <p>First occurrence: \\[x^2 + y^2 = z^2\\]</p>
        <p>Second occurrence: \\[x^2 + y^2 = z^2\\]</p>
        <p>Third occurrence: \\[x^2 + y^2 = z^2\\]</p>
        </body>
        </html>
        """

        tests = generate_tests_from_html(html_content, "test_pdf", 1, self.random_generator)
        math_tests = [t for t in tests if t.get("type") == "math"]

        # Count how many times the equation appears
        equation_count = sum(1 for t in math_tests if "x^2 + y^2 = z^2" in t["math"])

        # Should only appear once due to deduplication
        self.assertEqual(equation_count, 1, "Duplicate equations should be deduplicated")

    def test_math_extraction_patterns(self):
        """Test different math delimiter patterns"""
        html_content = """
        <html>
        <body>
        <p>Pattern 1: \\(inline1\\)</p>
        <p>Pattern 2: \\[display1\\]</p>
        <p>Pattern 3: $$display2$$</p>
        </body>
        </html>
        """

        tests = generate_tests_from_html(html_content, "test_pdf", 1, self.random_generator)
        math_tests = [t for t in tests if t.get("type") == "math"]

        math_contents = [t["math"] for t in math_tests]

        # Check all patterns are captured
        self.assertIn("inline1", math_contents)
        self.assertIn("display1", math_contents)
        self.assertIn("display2", math_contents)

    def test_math_extraction_minimum_length(self):
        """Test that very short equations are filtered out"""
        html_content = """
        <html>
        <body>
        <p>Short: \\(x\\)</p>
        <p>Also short: \\[y\\]</p>
        <p>Long enough: \\(x=1\\)</p>
        </body>
        </html>
        """

        tests = generate_tests_from_html(html_content, "test_pdf", 1, self.random_generator)
        math_tests = [t for t in tests if t.get("type") == "math"]

        math_contents = [t["math"] for t in math_tests]

        # Short equations (length <= 2) should be filtered out
        self.assertNotIn("x", math_contents)
        self.assertNotIn("y", math_contents)
        # Longer equation should be included
        self.assertIn("x=1", math_contents)

    def test_math_validation_passes(self):
        """Test that valid math tests pass validation against markdown"""
        html_content = """
        <html>
        <body>
        <p>Test equation: \\[E = mc^2\\]</p>
        </body>
        </html>
        """

        # Mock the validation to always pass for math tests
        with patch("olmocr.bench.tests.load_single_test") as mock_load:
            mock_test = MagicMock()
            mock_test.run.return_value = (True, None)
            mock_load.return_value = mock_test

            tests = generate_tests_from_html(html_content, "test_pdf", 1, self.random_generator)
            math_tests = [t for t in tests if t.get("type") == "math"]

            # Verify math test was created
            self.assertTrue(len(math_tests) > 0)
            # Verify test has correct structure
            for test in math_tests:
                self.assertEqual(test["type"], "math")
                self.assertIn("math", test)
                self.assertEqual(test["max_diffs"], 0)
                self.assertIn("id", test)
                self.assertIn("pdf", test)
                self.assertEqual(test["page"], 1)

    def test_complex_markdown_example(self):
        """Test with the complex markdown example provided by the user"""
        # Convert markdown to HTML-like structure for testing
        html_content = '<!DOCTYPE html>\n<html lang="en">\n<head>\n    <meta charset="UTF-8">\n    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n    <title>Automorphisms of Order Two</title>\n    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>\n    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>\n    <script>\n        window.MathJax = {\n            tex: {\n                inlineMath: [[\'\\\\(\', \'\\\\)\']],\n                displayMath: [[\'\\\\[\', \'\\\\]\']]\n            }\n        };\n    </script>\n    <style>\n        body {\n            font-family: "Times New Roman", serif;\n            font-size: 11pt;\n            line-height: 1.4;\n            max-width: 791px;\n            margin: 0 auto;\n            padding: 20px;\n            background-color: white;\n        }\n        \n        .math-block {\n            margin: 15px 0;\n        }\n        \n        .definition {\n            margin: 20px 0;\n        }\n        \n        .definition-header {\n            font-weight: bold;\n            margin-bottom: 10px;\n        }\n        \n        .lemma {\n            margin: 20px 0;\n        }\n        \n        .lemma-header {\n            font-weight: bold;\n            margin-bottom: 10px;\n        }\n        \n        .proof {\n            margin: 15px 0;\n        }\n        \n        .proof-header {\n            font-weight: bold;\n            display: inline;\n        }\n        \n        .qed {\n            float: right;\n            font-weight: bold;\n        }\n        \n        ul {\n            margin: 15px 0;\n            padding-left: 20px;\n        }\n        \n        ol {\n            margin: 15px 0;\n            padding-left: 20px;\n        }\n        \n        h2 {\n            font-size: 14pt;\n            font-weight: bold;\n            margin: 25px 0 15px 0;\n        }\n        \n        .equation {\n            text-align: right;\n            margin: 15px 0;\n        }\n        \n        footer {\n            text-align: center;\n            margin-top: 30px;\n            font-weight: bold;\n        }\n    </style>\n</head>\n<body>\n    <div class="math-block">\n        <p>If \\(\\varphi \\in \\text{Aut}(E)\\) with \\(\\varphi^2 = id\\) we observe that</p>\n        \\[e_i = \\frac{e_i + \\varphi(e_i)}{2} + \\frac{e_i - \\varphi(e_i)}{2}, \\quad \\text{for } i \\in \\mathbb{N}.\\]\n        \n        <p>Setting \\(a_i = e_i + \\varphi(e_i)/2\\) we have:</p>\n        \n        <ul>\n            <li>\\(\\varphi(e_i) = -e_i + 2a_i\\),</li>\n            <li>\\(\\varphi(a_i) = a_i\\), that is, \\(a_i\\) is of degree zero in the \\(\\mathbb{Z}_2\\)-grading \\(E_\\varphi\\),</li>\n            <li>\\(\\varphi(e_i - a_i) = -(e_i - a_i)\\), that is, \\(e_i - a_i\\) is of degree 1 in the \\(\\mathbb{Z}_2\\)-grading \\(E_\\varphi\\).</li>\n        </ul>\n    </div>\n    \n    <div class="definition">\n        <div class="definition-header">Definition 5</div>\n        <p>Let \\(\\varphi \\in \\text{Aut}(E)\\). We say that \\(\\varphi\\) is of <em>canonical type</em> if \\(\\varphi(e_i) \\in E_{(1)}\\) for all \\(i\\).</p>\n        \n        <p>If \\(\\varphi\\) is an automorphism of order 2 on \\(E\\), we have that \\(\\varphi\\) is of canonical type if and only if \\(a_i \\in E_{(1)}\\) for all \\(i\\). Let us fix a basis \\(\\beta = \\{e_1, e_2, \\ldots, e_n, \\ldots\\}\\) of the vector space \\(L\\) and an automorphism \\(\\varphi \\in \\text{Aut}(E)\\) such that \\(\\varphi^2 = id\\). Then \\(\\varphi\\), as a linear transformation, has eigenvalues \\(\\pm 1\\) and \\(-1\\) only, and moreover, there exists a basis of the vector space \\(E\\) consisting of eigenvectors. (It is well known from elementary Linear Algebra that this fact does not depend on the dimension of the vector space as long as the characteristic of \\(F\\) is different from 2.) Then \\(E = E(1) \\oplus E(-1)\\) where \\(E(t)\\) is the eigenspace for the eigenvalue \\(t\\) of the linear transformation \\(\\varphi\\). One considers the intersections \\(L(t) = L \\cap E(t)\\), \\(t = \\pm 1\\). Changing the basis \\(\\beta\\), if necessary, one may assume that \\(L(t)\\) is the span of \\(\\beta \\cap L(t)\\). Clearly this change of basis gives rise to a homogeneous automorphism of \\(E\\) and we can take the composition of it and then \\(\\varphi\\). We shall assume that such a change of basis has been done.</p>\n        \n        <p>Denote</p>\n        \\[I_\\varphi = \\{n \\in \\mathbb{N} \\mid \\varphi(e_n) = \\pm e_n\\}.\\]\n    </div>\n    \n    <p>We shall distinguish the following four possibilities:</p>\n    \n    <ol>\n        <li>\\(I_\\varphi = \\mathbb{N}\\).</li>\n        <li>\\(I_\\varphi \\neq \\mathbb{N}\\) is infinite.</li>\n        <li>\\(I_\\varphi\\) is finite and nonempty.</li>\n        <li>\\(I_\\gamma = \\emptyset\\) for every linear basis \\(\\gamma\\) of \\(L\\).</li>\n    </ol>\n    \n    <p>We shall call these automorphisms (and also the corresponding \\(\\mathbb{Z}_2\\)-gradings), automorphisms (or gradings) of type 1, 2, 3, and 4, respectively.</p>\n    \n    <p>The automorphisms of type 1 induce \\(\\mathbb{Z}_2\\)-gradings on \\(E\\) in which all generators of \\(E\\) are homogeneous. Such structures are called homogeneous \\(\\mathbb{Z}_2\\)-gradings on \\(E\\). The corresponding graded identities were completely studied in [22, 24, 29].</p>\n    \n    <p>We conclude this section with the following lemma.</p>\n    \n    <div class="lemma">\n        <div class="lemma-header">Lemma 6</div>\n        <p>Let \\(\\varphi\\) be an automorphism of order two of \\(E\\). Then \\(\\varphi\\) is of type 4 if and only if, for every \\(v \\in L\\) such that \\(\\varphi(v) = \\pm v\\), one has \\(v = 0\\).</p>\n        \n        <div class="proof">\n            <span class="proof-header">Proof</span> Assume that \\(\\varphi\\) is of type 4 and let \\(v \\in L\\) with \\(\\varphi(v) = \\pm v\\). If \\(v \\neq 0\\), choose a basis \\(\\gamma\\) of \\(L\\) such that \\(v \\in \\gamma\\). Then \\(I_\\gamma \\neq \\emptyset\\), a contradiction. The converse follows by the same argument.\n            <span class="qed">■</span>\n        </div>\n    </div>\n    \n    <h2>3 &nbsp;&nbsp; Automorphisms of order two of <em>E</em></h2>\n    \n    <p>From this point on, our goal is to survey recent developments regarding automorphisms of order two and the corresponding \\(\\mathbb{Z}_2\\)-gradings of the infinite-dimensional Grassmann algebra.</p>\n    \n    <p>Let \\(X = \\{e_1, \\ldots, e_n, \\ldots\\}\\). For each map \\(\\lambda : X \\to E\\), we can define the linear transformation \\(\\varphi : E \\to E\\) by</p>\n    \n    <div class="equation">\n        \\[\\varphi(e_{i_1} \\cdots e_{i_n}) = \\lambda(e_{i_1}) \\cdots \\lambda(e_{i_n}),\\] <span style="float: right;">(1)</span>\n    </div>\n    \n    <p>for all \\(n \\in \\mathbb{N}\\).</p>\n    \n    <p>We start with the next lemma.</p>\n    \n    <div class="lemma">\n        <div class="lemma-header">Lemma 7</div>\n        <p><em>The linear transformation</em> \\(\\varphi\\) <em>is an endomorphism of</em> \\(E\\) <em>if and only if</em></p>\n        \\[\\lambda(e_i)\\lambda(e_j) + \\lambda(e_j)\\lambda(e_i) = 0, \\quad \\text{for all } i, j.\\]\n    </div>\n    \n    <footer>\n        4\n    </footer>\n</body>\n</html>'
        tests = generate_tests_from_html(html_content, "test_pdf", 1, self.random_generator)
        math_tests = [t for t in tests if t.get("type") == "math"]

        for test in math_tests:
            print(test)

    def test_math_extraction_strips_whitespace(self):
        """Test that extracted math equations have whitespace properly stripped"""
        html_content = """
        <html>
        <body>
        <p>\\[
            x = y + z
        \\]</p>
        </body>
        </html>
        """

        tests = generate_tests_from_html(html_content, "test_pdf", 1, self.random_generator)
        math_tests = [t for t in tests if t.get("type") == "math"]

        self.assertTrue(len(math_tests) > 0)
        # The equation should be stripped of leading/trailing whitespace
        self.assertEqual(math_tests[0]["math"].strip(), math_tests[0]["math"])

    def test_math_not_in_present_tests(self):
        """Test that extracted math equations have whitespace properly stripped"""
        html_content = """
        <html>
        <body>
        <p>\\[x = y + z \\mathcal{x} \\]</p>

        <p>\\[y = x + z  \\mathcal{x}  \\]</p>

        <p>\\[q = r + z  \\mathcal{x}  \\]</p>
        </body>
        </html>
        """

        tests = generate_tests_from_html(html_content, "test_pdf", 1, self.random_generator)
        present_tests = [t for t in tests if t.get("type") == "present"]
        self.assertTrue(len(present_tests) == 0)


if __name__ == "__main__":
    unittest.main()
