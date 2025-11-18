import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from olmocr.synth.mine_html_templates import generate_tests_from_html


class TestFootnoteTestGeneration(unittest.TestCase):
    def setUp(self):
        self.random_gen = random.Random(42)
        self.pdf_id = "test_pdf"
        self.page_num = 1
        self.uuid_counter = 0
        uuid_patch = patch(
            "olmocr.synth.mine_html_templates.uuid.uuid4",
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
            self._hashable_tests(
                [
                    {
                        "pdf": "test_pdf_page1.pdf",
                        "page": 1,
                        "type": "footnote",
                        "marker": "1",
                        "max_diffs": 0,
                        "appears_before_marker": "Alpha with reference",
                    }
                ]
            ),
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
            self._hashable_tests(
                [
                    {
                        "pdf": "test_pdf_page1.pdf",
                        "page": 1,
                        "type": "footnote",
                        "marker": "1",
                        "max_diffs": 0,
                        "appears_before_marker": "Alpha with reference",
                        "appears_after_marker": "This is the",
                    }
                ]
            ),
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
            self._hashable_tests(
                [
                    {
                        "pdf": "test_pdf_page1.pdf",
                        "page": 1,
                        "type": "footnote",
                        "marker": "1",
                        "max_diffs": 0,
                        "appears_before_marker": "ends with marker",
                        "appears_after_marker": "Definition for footnote",
                    },
                    {
                        "pdf": "test_pdf_page1.pdf",
                        "page": 1,
                        "type": "footnote",
                        "marker": "2",
                        "max_diffs": 0,
                        "appears_before_marker": "paragraph carries marker",
                        "appears_after_marker": "Definition for footnote",
                    },
                    {
                        "pdf": "test_pdf_page1.pdf",
                        "page": 1,
                        "type": "footnote",
                        "marker": "3",
                        "max_diffs": 0,
                        "appears_before_marker": "a lone marker",
                    },
                ]
            ),
        )

    def test_cutoff(self):
        html_content = """<html><body>Your personal data is processed in accordance with Regulation (EU) No 2018/1725<sup><a href="#footnote1">1</a></sup> on the protection of individuals with regard to the processing of personal data by the Union institutions, bodies, offices and agencies and on the free movement of such data.</body></html>"""

        footnote_tests = self._generate_footnote_tests(html_content)

        self.assertEqual(len(footnote_tests), 1)

        self.assertSetEqual(
            footnote_tests,
            self._hashable_tests(
                [
                    {
                        "pdf": "test_pdf_page1.pdf",
                        "page": 1,
                        "type": "footnote",
                        "marker": "1",
                        "max_diffs": 0,
                        "appears_before_marker": "(EU) No 2018/1725",
                    },
                ]
            ),
        )

    def test_footnotes_in_footer(self):
        # Sometimes footnotes appear in a <footer> tag, and we like to remove text from footer tags and adjust how things work with it
        # So the idea here is to test that
        html_content = """
<!DOCTYPE html>

<html lang="en">
<head><meta content="ac4a05db236c9eee46a49a9545c64ea84923b5b7" name="olmocr_git_commit"/>
<meta charset="utf-8"/>
<meta content="width=725, height=1024" name="viewport"/>
<title>9.3 Principles of Lagrangian construction</title>
</head>
<body>
<header>
<span class="section-title">9.3 Principles of Lagrangian construction</span>
<span class="page-number">133</span>
</header>
<h3>9.3.2 A worked-out example: Vector and tensor fields</h3>
<p>
        The previous analysis can be easily extended to other types of fields. Consider for instance a vector field $A_\mu$ and a tensor field $B_{\mu\nu}$. What is the most general Lagrangian $\mathcal{L}(A, \partial A, B, \partial B)$ that can be constructed out of these two fields? To clarify the procedure, let me split the problem into several pieces
    </p>
<footer>
<div class="footnote">
<sup>3</sup>Quadratic actions give rise to linear equations of motion, where the superposition principle can be applied.
        </div>
<div class="footnote">
<sup>4</sup>In the same way that <i>physically</i> cannot be attributed to $\mathcal{L}$, we cannot make any claim about the physicality of $A_\mu$. Physicality might be attributed to the set $\{A_\mu\}$ of gauge-equivalent 4-potentials or to any gauge invariant attribute of that set, but not to its individual elements.
        </div>
<div class="footnote">
<sup>5</sup>It cannot be compensated by the transformation of the other (derivative) terms.
        </div>
</footer>
</body>
</html>
"""
        footnote_tests = self._generate_footnote_tests(html_content)

        print(footnote_tests)

        self.assertEqual(len(footnote_tests), 3)

    def test_footnotes_in_footer_ptags(self):
        # Sometimes footnotes appear in a <footer> tag, and we like to remove text from footer tags and adjust how things work with it
        # So the idea here is to test that
        html_content = """
<!DOCTYPE html>

<html lang="en">
<head><meta content="ac4a05db236c9eee46a49a9545c64ea84923b5b7" name="olmocr_git_commit"/>
<meta charset="utf-8"/>
<meta content="width=725, height=1024" name="viewport"/>
<title>9.3 Principles of Lagrangian construction</title>
</head>
<body>
<header>
<span class="section-title">9.3 Principles of Lagrangian construction</span>
<span class="page-number">133</span>
</header>
<h3>9.3.2 A worked-out example: Vector and tensor fields</h3>
<p>
        The previous analysis can be easily extended to other types of fields. Consider for instance a vector field $A_\mu$ and a tensor field $B_{\mu\nu}$. What is the most general Lagrangian $\mathcal{L}(A, \partial A, B, \partial B)$ that can be constructed out of these two fields? To clarify the procedure, let me split the problem into several pieces
    </p>
<footer>
<p class="footnote">
<sup>3</sup>Quadratic actions give rise to linear equations of motion, where the superposition principle can be applied.
        </p>
<p class="footnote">
<sup>4</sup>In the same way that <i>physically</i> cannot be attributed to $\mathcal{L}$, we cannot make any claim about the physicality of $A_\mu$. Physicality might be attributed to the set $\{A_\mu\}$ of gauge-equivalent 4-potentials or to any gauge invariant attribute of that set, but not to its individual elements.
        </p>
<p class="footnote">
<sup>5</sup>It cannot be compensated by the transformation of the other (derivative) terms.
        </p>
        <p class="footnote">Don't make an absense test</p>
</footer>
</body>
</html>
"""
        tests = generate_tests_from_html(
            html_content,
            self.pdf_id,
            self.page_num,
            self.random_gen,
            False,
        )

        self.assertEqual(len([test for test in tests if test["type"] == "footnote"]), 3)        

        for test in [test for test in tests if test["type"] == "absent"]:
            self.assertNotEqual(test["text"], "Don't make an absense test")

    def test_sup_tags_in_text(self):
        """Test that sup/sub tags are preserved in markdown but not in generated test text fields"""
        html = """
<!DOCTYPE html>

<html lang="en">
<head><meta content="dd47b244c4e5f53cbaf95c7a2464507c2182f2bb" name="olmocr_git_commit"/>
<meta charset="utf-8"/>
<meta content="width=774, height=1024" name="viewport"/>
<title>Management of Symptomatic Stenosis</title>
</head>
<body>
<header>
<span class="page-number">348</span>
<span class="running-header">D. Barros Casas et al. / Arch Bronconeumol. 2013;9(8):345–354</span>
</header>
<p class="figure-caption"><strong>Figure 3.</strong> Flowchart for the management of symptomatic stenosis. (*) Radial incisions, balloon dilation and topical application of mitomycin.</p>
<div class="two-column">
<div class="column">
<p>as an aid for other endoscopic techniques at various levels of the airway or as the sole technique in the case of simple, short stenoses that do not completely obstruct the airway lumen; this technique is well supported in the scientific literature.<sup>57</sup></p>
<p>Laser is only useful in small, narrow lesions with a reduced vertical length and stable cartilaginous skeleton, although it is widely and generally used with equally good results and low risk in the case of larger lesions. The decannulation rate is high, surgical time is reduced, and hospital stay is short-term.<sup>58</sup> For web-like stenosis, there is a variation of the technique that involves making radial incisions with the laser or with the electrocautery knife at 3, 9 and 12 o'clock before dilating.<sup>58–59</sup></p>
<p>The microdebrider has been shown to be effective in lesions with excessive granulation tissue.<sup>51–52</sup></p>
<p>Stenting is indicated in patients who do not respond to endoscopic dilation and are not candidates for surgical resection. It is important to remember that the stents indicated for this type of lesion must be easy to remove; at present, silicone stents are the most widely used, although there are also reports of the use of coated AERO hybrid nitinol stents. These are self-expanding and can be removed, and do not require rigid bronchoscopy for implantation.<sup>43,45</sup> Loss of cartilaginous support in the absence of extrinsic compression leads to migration of stents located in the subglottic region or proximal trachea. In these cases, external percutaneous fixation may be considered. Potential complications include skin infections around the external button.<sup>46,47</sup> Re-stenosis as a result of the repair process itself and stent obstruction are the main reasons for re-intervention.<sup>29,32</sup></p>
<p>The use of topical mitomycin is controversial, but together with radial laser incisions and balloon dilation it has some beneficial effect compared to placebo at 2–3 years<sup>60–62</sup> (Fig. 3).</p>
<p>Subglottic stenosis, mainly caused by intubation, deserves a special mention. The subglottic space refers to the section of the airway between the vocal cords and the lower traction of the cricoid cartilage, which is the narrowest section of the larynx and the only one surrounded by a complete ring of cartilage. Its narrow diameter, inconvenient location, surrounding tissue fragility of the coating mucosa and the tendency to form granulation tissue and scars from intubation, re-stenosis and failure to decannulate.<sup>51</sup> An incidence of subglottic stenosis secondary to prolonged intubation in children and adults ranging from 0.9% to 8.3% has been reported.<sup>52</sup> Management is a challenge involving various strategies that must be tailored to suit each patient. For non-concentric soft, membranous stenoses with sufficient cartilaginous support and a length of</p>
</div>
<div class="column">
<p>around one centimeter corresponding to Cotton-Meyer grades I and II, endoscopic techniques described above are used, with emphasis on the use of laser. The success rate is variable according to the literature, ranging between 40% and 94%.<sup>53</sup> Longer, hard, grade III and IV complex stenoses can be treated initially with endoscopic techniques, but in most cases, open reconstructive surgery will be required (surgical resection of the stenosed section, including several tracheal rings and the anterior cricoid ring, in addition to the lower half of the mucosa of the cricoid cartilage, followed by end-to-end anastomosis).<sup>54,54</sup></p>
<h2>Dynamic Airway Obstruction: Tracheobronchomalacia and Excessive Pars Membranacea Collapse</h2>
<p>TBM and excessive pars membranacea collapse occur in around 12% of patients with respiratory diseases.<sub>55</sub> In TBM, the proportion between cartilage and soft tissues is reduced from a normal ratio of 4:1 to 1:1.<sup>56</sup> While in excessive pars membranacea collapse, there is atrophy and a loss of myoelastic fibers.<sup>57</sup> TBM, in both its local and diffuse forms, can affect the trachea, bronchi or both structures. There are different ways of classifying the disease, but the functional classification (FEMOS) is the most comprehensive.<sup>58</sup> TBM may be asymptomatic, although it often produces cough, wheezing, stridor, dyspnea, recurrent infections, and on occasions, respiratory failure,<sup>59</sup> and therefore differential diagnosis is needed to rule out disease entities such as chronic obstructive pulmonary disease, asthma and bronchiectasis.<sup>61</sup> Respiratory function tests can help in the diagnosis of concomitant obstructive pulmonary disease, but they have limited application in the diagnosis of TBM, since results are normal in up to 21% of cases.<sup>62</sup> Accordingly, dynamic chest tomography and dynamic flexible bronchoscopy are often required for diagnosis.<sup>63,64</sup> (Fig. 1, image 3). This disease can be easily diagnosed by the performance of dynamic inhalation and exhalation maneuvers. In patients with diffuse TBM, a diagnostic test must be performed with silicone stent placement,<sup>65–67</sup> along with management of comorbidities. Patients who show improvement in their symptoms and respiratory function will be candidates for surgical or medical reconstruction by tracheobronchoplasty.<sup>55</sup> Patients who cannot undergo surgery due to their comorbidities, will be managed with a combination of symptomatic treatment and possible definitive stenting (Fig. 4).</p>
<p>Although non-invasive ventilation has been proposed as a possible treatment for TBM, its role appears to be restricted to the</p>
</div>
</div>
</body>
</html>"""

        tests = generate_tests_from_html(
                html,
                self.pdf_id,
                self.page_num,
                self.random_gen,
                False,
         )

        # Check that tests containing sup/sub tags are filtered out
        # The test generation should filter these out already
        for test in tests:
            if "text" in test:
                # These tags should have been filtered out during test generation
                self.assertNotIn("<sup>", test["text"])
                self.assertNotIn("</sup>", test["text"])
                self.assertNotIn("<sub>", test["text"])
                self.assertNotIn("</sub>", test["text"])

    def test_footnotes_footer_again(self):
        html = """<!DOCTYPE html>

<html lang="en">
<head><meta content="dd47b244c4e5f53cbaf95c7a2464507c2182f2bb" name="olmocr_git_commit"/>
<meta charset="utf-8"/>
<meta content="width=724, initial-scale=1.0" name="viewport"/>
<title>KPN Second Quarter 2017 Results</title>
</head>
<body>
<header>
<div class="logo"></div>
<div class="press-release-info">
<div class="label">Press release</div>
<div class="date">26 July 2017</div>
</div>
</header>
<main>
<h1>Second Quarter 2017 Results</h1>
<h2>Highlights</h2>
<ul>
<li>Fixed-mobile convergence continues to deliver strong results in Consumer
                <ul>
<li>More than 60% of KPN brand postpaid base in fixed-mobile bundles (Q2 2016: 51%)</li>
<li>+8k broadband net adds, +25k IPTV net adds, and +9k postpaid net adds<sup>§</sup> driven by the high value KPN brand</li>
<li>Postpaid ARPU stable at EUR 26, ARPU per household increased by 5.0% y-on-y to EUR 42</li>
<li>Further improvement in customer satisfaction in Consumer; NPS +13 (Q2 2016: +9)</li>
</ul>
</li>
<li>Progress with Business transformation
                <ul>
<li>SME: fixed-mobile bundling on track, +39k multi play net adds driven by uptake KPN ÉÉN; positioning for growth in IT through up- and cross-sell</li>
<li>LE &amp; Corporate: focus on value in competitive mobile-only market; growth in IT related services, Internet of Things and Security</li>
<li>Dedicated to further improve customer satisfaction in Business; NPS -6 (Q2 2016: -9)</li>
</ul>
</li>
<li>Second wave Simplification program delivered ~EUR 65m run-rate savings by end Q2 2017</li>
</ul>
<h2>Key figures<sup>*</sup> (from continuing operations)</h2>
<div class="table-header">Group financials (unaudited)</div>
<div class="table-header">(in EUR m, unless stated otherwise)</div>
<table>
<thead>
<tr>
<th></th>
<th>Q2 2017</th>
<th>Q2 2016</th>
<th>Δ y-on-y</th>
<th>YTD 2017</th>
<th>YTD 2016</th>
<th>Δ y-on-y</th>
</tr>
</thead>
<tbody>
<tr>
<td>Revenues</td>
<td>1,631</td>
<td>1,676</td>
<td>-2.7%</td>
<td>3,279</td>
<td>3,365</td>
<td>-2.6%</td>
</tr>
<tr>
<td><strong>Adjusted revenues**</strong></td>
<td><strong>1,623</strong></td>
<td><strong>1,676</strong></td>
<td><strong>-3.2%</strong></td>
<td><strong>3,271</strong></td>
<td><strong>3,365</strong></td>
<td><strong>-2.8%</strong></td>
</tr>
<tr>
<td><strong>Adjusted revenues The Netherlands**</strong></td>
<td><strong>1,485</strong></td>
<td><strong>1,486</strong></td>
<td><strong>-2.1%</strong></td>
<td><strong>2,955</strong></td>
<td><strong>2,989</strong></td>
<td><strong>-1.8%</strong></td>
</tr>
<tr>
<td>EBITDA</td>
<td>587</td>
<td>579</td>
<td>1.4%</td>
<td>1,143</td>
<td>1,138</td>
<td>0.4%</td>
</tr>
<tr>
<td><strong>Adjusted EBITDA**</strong></td>
<td><strong>601</strong></td>
<td><strong>592</strong></td>
<td><strong>1.5%</strong></td>
<td><strong>1,185</strong></td>
<td><strong>1,160</strong></td>
<td><strong>2.2%</strong></td>
</tr>
<tr class="indent-row">
<td>Adjusted EBITDA margin The Netherlands</td>
<td>41.0%</td>
<td>39.5%</td>
<td>+150bps</td>
<td>40.1%</td>
<td>38.5%</td>
<td>+160bps</td>
</tr>
<tr>
<td>Operating profit (EBIT)</td>
<td>232</td>
<td>205</td>
<td>13%</td>
<td>433</td>
<td>346</td>
<td>25%</td>
</tr>
<tr>
<td>Profit for the period (net profit)</td>
<td>151</td>
<td>162</td>
<td>-18%</td>
<td>283</td>
<td>210</td>
<td>35%</td>
</tr>
<tr>
<td>Capex</td>
<td>238</td>
<td>312</td>
<td>-24%</td>
<td>504</td>
<td>630</td>
<td>-20%</td>
</tr>
<tr>
<td><strong>Operating free cash flow</strong></td>
<td><strong>363</strong></td>
<td><strong>280</strong></td>
<td><strong>30%</strong></td>
<td><strong>681</strong></td>
<td><strong>530</strong></td>
<td><strong>28%</strong></td>
</tr>
<tr>
<td><strong>Free cash flow</strong></td>
<td><strong>296</strong></td>
<td><strong>254</strong></td>
<td><strong>17%</strong></td>
<td><strong>331</strong></td>
<td><strong>214</strong></td>
<td><strong>55%</strong></td>
</tr>
</tbody>
</table>
<div class="footnote">* All non-IFRS terms are explained in the safe harbor section</div>
<div class="footnote">** Adjusted revenues and adjusted EBITDA reconciliations to be found on page 8 and 9</div>
<h2>Financial performance</h2>
<div class="financial-performance">
<ul>
<li>Adjusted revenues for The Netherlands were 2.1% lower y-on-y in Q2 2017. Customer base growth and higher ARPU per household led to revenue growth in Consumer, but this was offset by lower Business and Wholesale revenues. Revenues in Business were impacted by migrations and rationalization, and price pressure in mobile, while Wholesale saw lower revenues from MVNOs and international traffic</li>
<li>Adjusted EBITDA was 1.5% higher y-on-y in Q2 2017 due to improved operational efficiency as a result of simplification and lower subscriber retention and acquisition costs</li>
<li>Operating profit increased by 13% y-on-y in Q2 2017 mainly driven by EUR 19m lower amortization charges. Net profit increased by 18% y-on-y to EUR 191m in Q2 2017</li>
<li>Capex in H1 2017 was 20% lower y-on-y, mainly due to different intrayear phasing compared to last year</li>
<li>Free cash flow (excl. TEFD dividend) of EUR 261m in H1 2017 was EUR 157m higher compared to the same period last year, mainly driven by lower Capex and less interest paid</li>
</ul>
</div>
</main>
<footer>
<div class="footer-content">
<div class="footnote"><sup>§</sup> Adjusted for 15k migration to Business</div>
<div>KPN Management Report Q2 2017<span style="margin-left: 30px;">1</span></div>
</div>
</footer>
</body>
</html>"""

        tests = generate_tests_from_html(
                html,
                self.pdf_id,
                self.page_num,
                self.random_gen,
                False,
         )
        
        print(tests)


if __name__ == "__main__":
    unittest.main()
