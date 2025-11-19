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
    """Test superscript and subscript preservation as HTML tags in html_to_markdown_with_frontmatter"""

    def test_basic_superscripts(self):
        """Test basic superscript preservation"""
        html = """
        <html>
        <body>
            <p>x<sup>2</sup> + y<sup>3</sup> = z<sup>4</sup></p>
            <p>10<sup>9</sup> is a billion</p>
        </body>
        </html>
        """
        result = html_to_markdown_with_frontmatter(html)

        # Check that superscripts are preserved as HTML tags
        self.assertIn("x<sup>2</sup>", result)
        self.assertIn("y<sup>3</sup>", result)
        self.assertIn("z<sup>4</sup>", result)
        self.assertIn("10<sup>9</sup>", result)

    def test_basic_subscripts(self):
        """Test basic subscript preservation"""
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

        # Check that subscripts are preserved as HTML tags
        self.assertIn("H<sub>2</sub>O", result)
        self.assertIn("CO<sub>2</sub>", result)
        self.assertIn("X<sub>n</sub>", result)

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

        # Check mixed preserved tags
        self.assertIn("x<sup>2</sup>", result)
        self.assertIn("H<sub>2</sub>O<sup>+</sup>", result)
        self.assertIn("Ca<sup>2+</sup>", result)
        self.assertIn("SO<sub>4</sub><sup>2-</sup>", result)

    def test_special_characters(self):
        """Test special character preservation"""
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

        # Check special characters are preserved
        self.assertIn("(x+y)<sup>n</sup>", result)
        self.assertIn("f<sub>(x)</sub>", result)
        self.assertIn("OH<sup>-</sup>", result)
        self.assertIn("H<sup>+</sup>", result)
        self.assertIn("a<sub>i</sub>", result)
        self.assertIn("b<sup>i</sup>", result)

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

        # Tables should be preserved as HTML with sub/sup tags
        self.assertIn("<table>", result)

        # Check if tags are preserved in table cells
        self.assertIn("<sub>2</sub>", result)
        self.assertIn("<sub>4</sub><sup>2-</sup>", result)

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

        # Check tags are preserved in nested structures
        self.assertIn("mc<sup>2</sup>", result)
        self.assertIn("x<sup>1</sup>", result)
        self.assertIn("x<sub>2</sub>", result)

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

        # Also check the tags are preserved
        self.assertIn("x<sup>2</sup>", result)

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

        # All characters should be preserved as HTML tags
        self.assertIn("H<sub>2</sub>SO<sub>4</sub>", result)
        # Asterisk should be preserved in sup tag
        self.assertIn("note<sup>*</sup>", result)

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

        # Empty tags should be preserved (even if empty)
        self.assertIn("z<sup>2</sup>", result)
        # Empty tags might be preserved or removed - just check text is there
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

        # Check complex nested expressions are preserved as tags
        self.assertIn("x<sub>1</sub>", result)
        self.assertIn("x<sub>2</sub>", result)
        self.assertIn("r<sup>2</sup>", result)
        self.assertIn("a<sub>0</sub>", result)
        self.assertIn("a<sub>1</sub>", result)
        self.assertIn("a<sub>2</sub>", result)
        self.assertIn("a<sub>n</sub>", result)
        self.assertIn("x<sup>n</sup>", result)

    def test_non_numerical_footnote(self):
        """Test a complex mathematical expression"""
        html = """
        <html>
        <body>
            <p>This is a sentence<sup>wow</sup></p>
            <br/><br/>
            <p><sup>wow</sup> This is a footnote</p>
        </body>
        </html>
        """
        result = html_to_markdown_with_frontmatter(html)

        self.assertIn("<sup>wow</sup>", result)

    def test_line_numbers_markdown(self):
        html = """
<!DOCTYPE html>

<html lang="en">
<head><meta content="d1fdb60869ea7e1e0c48228f80477b67661e7654" name="olmocr_git_commit"/>
<meta charset="utf-8"/>
<meta content="width=792, height=1024" name="viewport"/>
<title>bioRxiv preprint</title>
</head>
<body>
<header>
        bioRxiv preprint doi: <a href="https://doi.org/10.1101/2020.10.16.342121">https://doi.org/10.1101/2020.10.16.342121</a>; this version posted October 19, 2020. The copyright holder for this preprint<br/>
        (which was not certified by peer review) is the author/funder. All rights reserved. No reuse allowed without permission.
    </header>
<div class="content">
<div class="line">
<div class="line-number">67</div>
<div class="line-text">to IIA site (Atukeren, Aydin, Uslu, Gumustas, &amp; Cakatay, 2010), however, IIIA site was also</div>
</div>
<div class="line">
<div class="line-number">68</div>
<div class="line-text">considered (Suji et al., 2008).</div>
</div>
<div class="line">
<div class="line-number">69</div>
<div class="line-text">Having in mind that DHLA is a very potent antioxidant and its use can alleviate a number of</div>
</div>
<div class="line">
<div class="line-number">70</div>
<div class="line-text">conditions related to oxidative stress, it seemed relevant to elucidate its mode of interaction with</div>
</div>
<div class="line">
<div class="line-number">71</div>
<div class="line-text">HSA, a universal transporter in the circulation. The properties of this interaction, are still</div>
</div>
<div class="line">
<div class="line-number">72</div>
<div class="line-text">unknown and undefined, so the present study aimed to investigate characteristics of the DHLA-</div>
</div>
<div class="line">
<div class="line-number">73</div>
<div class="line-text">HSA binding in detail, by using spectroscopic and molecular docking approach.</div>
</div>
<div class="line">
<div class="line-number">74</div>
<div class="line-text"></div>
</div>
<div class="section-heading">MATERIALS AND METHODS</div>
<div class="line">
<div class="line-number">75</div>
<div class="line-text"><h2>Materials</h2></div>
</div>
<div class="line">
<div class="line-number">76</div>
<div class="line-text">All chemicals used were of analytical grade and were purchased from Sigma (Burlington,</div>
</div>
<div class="line">
<div class="line-number">77</div>
<div class="line-text">Massachusetts, USA). Stock solution of HSA, purchased from Sigma (A-1653) and used without</div>
</div>
<div class="line">
<div class="line-number">78</div>
<div class="line-text">additional purification, was made by dissolving HSA in 10 mM PBS, pH 7.4. The concentration</div>
</div>
<div class="line">
<div class="line-number">79</div>
<div class="line-text">of HSA was determined by using bicinchoninic acid (BCA) assay kit (Thermo Fisher Scientific,</div>
</div>
<div class="line">
<div class="line-number">80</div>
<div class="line-text">Waltham, Massachusetts, USA). Stock solution (5 mM) of DHLA was prepared by suspending</div>
</div>
<div class="line">
<div class="line-number">81</div>
<div class="line-text">DHLA in 10 mM PBS and then adding a small volume of 1 M NaOH until full clarification of</div>
</div>
<div class="line">
<div class="line-number">82</div>
<div class="line-text">solution was reached (Perricone et al., 1999). Trypsin was purchased from the Institute Torlak</div>
</div>
<div class="line">
<div class="line-number">83</div>
<div class="line-text">(Belgrade, Serbia) as a 0.25 % solution. All experiments were performed in triplicate at room</div>
</div>
<div class="line">
<div class="line-number">84</div>
<div class="line-text">temperature, using 10 mM PBS, pH 7.4, unless otherwise stated.</div>
</div>
<div class="line">
<div class="line-number">85</div>
<div class="line-text"><h2>Spectrofluorometric analysis of HSA-DHLA complex formation</h2></div>
</div>
<div class="line">
<div class="line-number">86</div>
<div class="line-text">Binding constant (Ka) of HSA-DHLA complex was determined by recording the quenching of</div>
</div>
<div class="line">
<div class="line-number">87</div>
<div class="line-text">intrinsic fluorescence emission of HSA (0.4 μM) in the presence of increasing concentrations of</div>
</div>
<div class="line">
<div class="line-number">88</div>
<div class="line-text">DHLA (from 4 to 35 μM) at 37 °C. Fluorescence spectra were recorded using FluoroMax®-4</div>
</div>
<div class="line">
<div class="line-number">89</div>
<div class="line-text">spectrofluorometer (Horiba Scientific, Japan). HSA was excited at 280 nm and emission spectra</div>
</div>
<div class="line">
<div class="line-number">90</div>
<div class="line-text">were recorded in the range from 290 to 450 nm. Each spectrum was corrected for the emission of</div>
</div>
<div class="line">
<div class="line-number">91</div>
<div class="line-text">the control that contained only DHLA at particular concentration. The change of the emission</div>
</div>
<div class="line">
<div class="line-number">92</div>
<div class="line-text">intensity at 338 nm (HSA emission maximum) was used for the calculation of the binding</div>
</div>
<div class="line">
<div class="line-number">93</div>
<div class="line-text">constant. Emission intensity measured for HSA was first corrected for the small inner filter effect</div>
</div>
<div class="line">
<div class="line-number">94</div>
<div class="line-text">of DHLA using the equation:</div>
</div>
<div class="line">
<div class="line-number">95</div>
<div class="line-text">
<div class="equation">F<sub>c</sub> = F<sub>0</sub> × 10<sup>(Aex+Aem)/2</sup></div>
</div>
</div>
<div class="line">
<div class="line-number">96</div>
<div class="line-text">where Fc is corrected fluorescence, F<sub>0</sub> is measured fluorescence, Aex and Aem are absorbances</div>
</div>
<div class="line">
<div class="line-number">97</div>
<div class="line-text">at excitation and emission wavelengths which are 290 nm and 338 nm, respectively.</div>
</div>
<div class="line">
<div class="line-number">98</div>
<div class="line-text">Using corrected fluorescence, binding constant between HSA and DHLA was calculated using</div>
</div>
<div class="line">
<div class="line-number">99</div>
<div class="line-text">the following equation:</div>
</div>
</div>
<footer>
        3
    </footer>
</body>
</html>"""

        result = html_to_markdown_with_frontmatter(html)

        self.assertNotIn("96", result)
        self.assertNotIn("97", result)
        self.assertNotIn("98", result)
        self.assertNotIn("82", result)

    def test_svg_in_markdown(self):
        html = """<!DOCTYPE html>

<html lang="en">
<head><meta content="d1fdb60869ea7e1e0c48228f80477b67661e7654" name="olmocr_git_commit"/>
<meta charset="utf-8"/>
<meta content="width=device-width, initial-scale=1.0" name="viewport"/>
<title>Survey Results - Q6</title>
</head>
<body>
<div class="question-header">
        Q6. What, if anything, do you consider are the main barriers to the trading performance of your business?
    </div>
<div class="response-line total-row">
<div class="response-text"></div>
<div class="response-values">38   (100.00%)</div>
</div>
<div class="other-section">
<div class="other-header">Other:</div>
<div class="other-text">
            Restrictions relating to listing of building and any alterations/modifications to make it more usable [1]; Lack of decent public transport - it is impossible for staff out of the area to reach Rothbury for 9am [1]
        </div>
</div>
<div class="not-answered">
        Not Answered:     3
    </div>
<div class="chart-container">
<svg height="175" width="475" xmlns="http://www.w3.org/2000/svg">
<!-- Y-axis labels -->
<text font-family="Arial" font-size="8" x="10" y="15">35.0%</text>
<text font-family="Arial" font-size="8" x="10" y="38">30.0%</text>
<text font-family="Arial" font-size="8" x="10" y="62">25.0%</text>
<text font-family="Arial" font-size="8" x="10" y="85">20.0%</text>
<text font-family="Arial" font-size="8" x="10" y="108">15.0%</text>
<text font-family="Arial" font-size="8" x="10" y="131">10.0%</text>
<text font-family="Arial" font-size="8" x="15" y="154">5.0%</text>
<text font-family="Arial" font-size="8" x="15" y="170">0.0%</text>
<!-- Bars (heights scaled to fit - max 35% = ~155px height) -->
<!-- a: 28.95% -->
<rect fill="#6666ff" height="126" stroke="#000" stroke-width="0.5" width="20" x="55" y="42"></rect>
<!-- b: 5.26% -->
<rect fill="#6666ff" height="23" stroke="#000" stroke-width="0.5" width="20" x="80" y="145"></rect>
<!-- c: 2.63% -->
<rect fill="#6666ff" height="11" stroke="#000" stroke-width="0.5" width="20" x="105" y="157"></rect>
<!-- d: 18.42% -->
<rect fill="#6666ff" height="83" stroke="#000" stroke-width="0.5" width="20" x="130" y="85"></rect>
<!-- e: 0% (tiny bar) -->
<rect fill="#6666ff" height="2" stroke="#000" stroke-width="0.5" width="20" x="155" y="166"></rect>
<!-- f: 2.63% -->
<rect fill="#6666ff" height="11" stroke="#000" stroke-width="0.5" width="20" x="180" y="157"></rect>
<!-- g: 7.89% -->
<rect fill="#6666ff" height="36" stroke="#000" stroke-width="0.5" width="20" x="205" y="132"></rect>
<!-- h: 5.26% -->
<rect fill="#6666ff" height="23" stroke="#000" stroke-width="0.5" width="20" x="230" y="145"></rect>
<!-- i: 0% -->
<!-- j: 0% -->
<!-- k: 7.89% -->
<rect fill="#6666ff" height="36" stroke="#000" stroke-width="0.5" width="20" x="255" y="132"></rect>
<!-- l: 5.26% -->
<rect fill="#6666ff" height="23" stroke="#000" stroke-width="0.5" width="20" x="280" y="145"></rect>
<!-- m: 7.89% -->
<rect fill="#6666ff" height="36" stroke="#000" stroke-width="0.5" width="20" x="305" y="132"></rect>
<!-- n: 5.26% -->
<rect fill="#6666ff" height="23" stroke="#000" stroke-width="0.5" width="20" x="330" y="145"></rect>
<!-- o: 2.63% -->
<!-- p: 0% -->
<!-- q: 2.63% -->
<rect fill="#6666ff" height="11" stroke="#000" stroke-width="0.5" width="20" x="380" y="157"></rect>
<!-- X-axis labels -->
<text font-family="Arial" font-size="8" x="62" y="175">a</text>
<text font-family="Arial" font-size="8" x="87" y="175">b</text>
<text font-family="Arial" font-size="8" x="112" y="175">c</text>
<text font-family="Arial" font-size="8" x="137" y="175">d</text>
<text font-family="Arial" font-size="8" x="162" y="175">e</text>
<text font-family="Arial" font-size="8" x="187" y="175">f</text>
<text font-family="Arial" font-size="8" x="212" y="175">g</text>
<text font-family="Arial" font-size="8" x="237" y="175">h</text>
<text font-family="Arial" font-size="8" x="262" y="175">i</text>
<text font-family="Arial" font-size="8" x="287" y="175">j</text>
<text font-family="Arial" font-size="8" x="312" y="175">k</text>
<text font-family="Arial" font-size="8" x="337" y="175">l</text>
<text font-family="Arial" font-size="8" x="362" y="175">m</text>
<text font-family="Arial" font-size="8" x="387" y="175">n</text>
<!-- Axis lines -->
<line stroke="#000" stroke-width="1" x1="50" x2="450" y1="168" y2="168"></line>
<line stroke="#000" stroke-width="1" x1="50" x2="50" y1="10" y2="168"></line>
</svg>
</div>
</body>
</html>"""

        result = html_to_markdown_with_frontmatter(html)

        self.assertNotIn("a\nb\nc", result)
        self.assertIn("Graphic Placeholder", result)


if __name__ == "__main__":
    unittest.main()
