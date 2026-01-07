import unittest

from olmocr.bench.tests import (
    parse_html_tables,
    parse_markdown_tables,
)


class TestParseHtmlTables(unittest.TestCase):
    def test_basic_table(self):
        data = parse_html_tables(
            """
                    <table border="1">
                        <thead>
                            <tr>
                                <th></th>
                                <th>ArXiv</th>
                                <th>Old<br>scans<br>math</th>
                                <th>Tables</th>
                                <th>Old<br>scans</th>
                                <th>Headers<br>&<br>footers</th>
                                <th>Multi<br>column</th>
                                <th>Long<br>tiny<br>text</th>
                                <th>Base</th>
                                <th>Overall</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Mistral OCR API</td>
                                <td>77.2</td>
                                <td>67.5</td>
                                <td>60.6</td>
                                <td>29.3</td>
                                <td>93.6</td>
                                <td>71.3</td>
                                <td>77.1</td>
                                <td>99.4</td>
                                <td>72.0±1.1</td>
                            </tr>
                        </tbody></table>"""
        )[0]

        print(data)
        self.assertTrue(data.is_rectangular)

        self.assertEqual(data.cell_text[0, 0], "")
        self.assertEqual(data.cell_text[0, 1], "ArXiv")

        self.assertEqual(data.left_relations[0, 0], set())
        self.assertEqual(data.up_relations[0, 0], set())

        self.assertEqual(data.left_relations[0, 1], {(0, 0)})
        self.assertEqual(data.up_relations[1, 0], {(0, 0)})

        self.assertEqual(data.heading_cells, {(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9)})

        self.assertEqual(data.top_heading_relations(1, 3), {(0, 3)})

        # If there are no left headings defined, then the left most column is considered the left heading
        print(data.left_heading_relations(1, 3))
        self.assertEqual(data.left_heading_relations(1, 3), {(1, 0)})

    def test_multiple_top_headings(self):
        data = parse_html_tables(
            """
                    <table border="1">
                        <thead>
                            <tr>
                               <th colspan="2">Fruit Costs in Unittest land</th>
                            </tr>
                            <tr>
                                <th>Fruit Type</th>
                                <th>Cost</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Apples</td>
                                <td>$1.00</td>
                            </tr>
                            <tr>
                                <td>Oranges</td>
                                <td>$2.00</td>
                            </tr>                                 
                        </tbody></table>"""
        )[0]

        print(data)

        self.assertEqual(data.cell_text[0, 0], "Fruit Costs in Unittest land")
        self.assertEqual(data.cell_text[1, 0], "Fruit Type")
        self.assertEqual(data.cell_text[1, 1], "Cost")
        self.assertEqual(data.cell_text[2, 0], "Apples")
        self.assertEqual(data.cell_text[2, 1], "$1.00")
        self.assertEqual(data.cell_text[3, 0], "Oranges")
        self.assertEqual(data.cell_text[3, 1], "$2.00")

        self.assertEqual(data.up_relations[1, 0], {(0, 0)})
        self.assertEqual(data.up_relations[1, 1], {(0, 0)})

        self.assertEqual(data.up_relations[2, 0], {(1, 0)})
        self.assertEqual(data.up_relations[2, 1], {(1, 1)})

        self.assertEqual(data.top_heading_relations(1, 0), {(0, 0)})
        self.assertEqual(data.top_heading_relations(1, 1), {(0, 0)})

        self.assertEqual(data.top_heading_relations(2, 0), {(0, 0), (1, 0)})
        self.assertEqual(data.top_heading_relations(2, 1), {(0, 0), (1, 1)})

        self.assertTrue(data.is_rectangular)

    def test_extra_colspans(self):
        data = parse_html_tables(
            """
                    <table border="1">
                        <thead>
                            <tr>
                               <th colspan="3">Fruit Costs in Unittest land</th>
                            </tr>
                            <tr>
                                <th>Fruit Type</th>
                                <th>Cost</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Apples</td>
                                <td>$1.00</td>
                                <td>Hi</td>
                            </tr>
                            <tr>
                                <td>Oranges</td>
                                <td>$2.00</td>
                            </tr>                                 
                        </tbody></table>"""
        )[0]

        self.assertFalse(data.is_rectangular)

    def test_extra_rowspans(self):
        data = parse_html_tables(
            """
                    <table border="1">
                        <thead>
                            <tr>
                               <th colspan="2">Fruit Costs in Unittest land</th>
                            </tr>
                            <tr>
                                <th>Fruit Type</th>
                                <th>Cost</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td rowspan="2">Apples</td>
                                <td>$1.00</td>
                            </tr>
                            <tr>
                                <td>Oranges</td>
                                <td>$2.00</td>
                            </tr>                                 
                        </tbody></table>"""
        )[0]

        self.assertFalse(data.is_rectangular)

    def test_extra_rowspans_okay(self):
        data = parse_html_tables(
            """
       <table border="1">
                        <thead>
                            <tr>
                               <th colspan="2">Fruit Costs in Unittest land</th>
                            </tr>
                            <tr>
                                <th>Fruit Type</th>
                                <th>Cost</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Apples</td>
                                <td>$1.00</td>
                            </tr>
                            <tr>
                                <td rowspan="2">Oranges</td>
                                <td>$2.00</td>
                            </tr> 
                            <tr><td>a</td></tr>                                
                        </tbody></table>"""
        )[0]

        self.assertTrue(data.is_rectangular)

    def test_4x4_table_with_spans(self):
        """Test a 4x4 table with various row spans and column spans"""
        data = parse_html_tables(
            """
                    <table border="1">
                        <thead>
                            <tr>
                                <th>Header 1</th>
                                <th colspan="2">Header 2-3</th>
                                <th>Header 4</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td rowspan="2">Cell A (spans 2 rows)</td>
                                <td>Cell B</td>
                                <td>Cell C</td>
                                <td rowspan="2">Cell D (spans 2 rows)</td>
                            </tr>
                            <tr>
                                <td colspan="2">Cell E-F (spans 2 cols)</td>
                            </tr>
                            <tr>
                                <td>Cell G</td>
                                <td colspan="3">Cell H-I-J (spans 3 cols)</td>
                            </tr>
                        </tbody>
                    </table>"""
        )[0]

        print(data)
        self.assertTrue(data.is_rectangular)

        # Test header row
        self.assertEqual(data.cell_text[0, 0], "Header 1")
        self.assertEqual(data.cell_text[0, 1], "Header 2-3")

        self.assertNotIn((0, 2), data.cell_text)  # colspan=2, so that next cell is empty
        self.assertEqual(data.cell_text[0, 3], "Header 4")

        # Test first body row
        self.assertEqual(data.cell_text[1, 0], "Cell A (spans 2 rows)")
        self.assertEqual(data.cell_text[1, 1], "Cell B")
        self.assertEqual(data.cell_text[1, 2], "Cell C")
        self.assertEqual(data.cell_text[1, 3], "Cell D (spans 2 rows)")

        # Test second body row
        self.assertNotIn((2, 0), data.cell_text)
        self.assertEqual(data.cell_text[2, 1], "Cell E-F (spans 2 cols)")

        # Test third body row
        self.assertEqual(data.cell_text[3, 0], "Cell G")
        self.assertEqual(data.cell_text[3, 1], "Cell H-I-J (spans 3 cols)")

        # Test heading cells
        self.assertEqual(data.heading_cells, {(0, 0), (0, 1), (0, 3)})

        self.assertEqual(data.left_heading_relations(0, 0), set())
        self.assertEqual(data.left_heading_relations(1, 0), set())
        self.assertEqual(data.left_heading_relations(2, 0), set())
        self.assertEqual(data.left_heading_relations(3, 0), set())

        self.assertEqual(data.top_heading_relations(0, 0), set())
        self.assertEqual(data.top_heading_relations(0, 1), set())
        self.assertEqual(data.top_heading_relations(0, 2), set())
        self.assertEqual(data.top_heading_relations(0, 3), set())

        self.assertEqual(data.left_heading_relations(1, 1), {(1, 0)})
        self.assertEqual(data.left_heading_relations(1, 2), {(1, 0)})
        self.assertEqual(data.left_heading_relations(1, 3), {(1, 0)})

        self.assertEqual(data.top_heading_relations(3, 1), {(0, 1), (0, 3)})

    def test_complex_multi_level_headers(self):
        """Test a table with multiple levels of headers and complex spanning"""
        data = parse_html_tables(
            """
                    <table border="1">
                        <thead>
                            <tr>
                                <th rowspan="3"></th>
                                <th colspan="6">Main Category</th>
                            </tr>
                            <tr>
                                <th colspan="3">Sub Category A</th>
                                <th colspan="3">Sub Category B</th>
                            </tr>
                            <tr>
                                <th>A1</th>
                                <th>A2</th>
                                <th>A3</th>
                                <th>B1</th>
                                <th>B2</th>
                                <th>B3</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Row 1</td>
                                <td>10</td>
                                <td>20</td>
                                <td>30</td>
                                <td>40</td>
                                <td>50</td>
                                <td>60</td>
                            </tr>
                            <tr>
                                <td>Row 2</td>
                                <td>15</td>
                                <td>25</td>
                                <td>35</td>
                                <td>45</td>
                                <td>55</td>
                                <td>65</td>
                            </tr>
                        </tbody>
                    </table>"""
        )[0]

        print("\n=== Complex Multi-Level Headers Test ===")
        print(data)

        self.assertTrue(data.is_rectangular)

        # Test the three-level header structure
        self.assertEqual(data.cell_text[0, 0], "")  # Empty corner cell
        self.assertEqual(data.cell_text[0, 1], "Main Category")

        self.assertEqual(data.cell_text[1, 1], "Sub Category A")
        self.assertEqual(data.cell_text[1, 4], "Sub Category B")

        self.assertEqual(data.cell_text[2, 1], "A1")
        self.assertEqual(data.cell_text[2, 2], "A2")
        self.assertEqual(data.cell_text[2, 3], "A3")
        self.assertEqual(data.cell_text[2, 4], "B1")
        self.assertEqual(data.cell_text[2, 5], "B2")
        self.assertEqual(data.cell_text[2, 6], "B3")

        # Test data rows
        self.assertEqual(data.cell_text[3, 0], "Row 1")
        self.assertEqual(data.cell_text[3, 1], "10")
        self.assertEqual(data.cell_text[4, 0], "Row 2")
        self.assertEqual(data.cell_text[4, 1], "15")

        # Test heading cells - all header rows should be marked as heading cells
        expected_heading_cells = {
            (0, 0),
            (0, 1),  # First header row
            (1, 1),
            (1, 4),  # Second header row
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 5),
            (2, 6),  # Third header row
        }
        self.assertEqual(data.heading_cells, expected_heading_cells)

        # Test top heading relations for data cells
        # Cell (3, 1) should have relations to all three levels of headers above it
        top_relations_3_1 = data.top_heading_relations(3, 1)
        self.assertIn((0, 1), top_relations_3_1)  # Main Category
        self.assertIn((1, 1), top_relations_3_1)  # Sub Category A
        self.assertIn((2, 1), top_relations_3_1)  # A1

        # Cell (3, 4) should relate to headers in the B column
        top_relations_3_4 = data.top_heading_relations(3, 4)
        self.assertIn((0, 1), top_relations_3_4)  # Main Category
        self.assertIn((1, 4), top_relations_3_4)  # Sub Category B
        self.assertIn((2, 4), top_relations_3_4)  # B1

        # Test left heading relations
        self.assertEqual(data.left_heading_relations(3, 1), {(3, 0)})  # Row 1
        self.assertEqual(data.left_heading_relations(4, 1), {(4, 0)})  # Row 2

    def test_left_headers_with_row_spans(self):
        """Test a table with left-side headers that have row spans"""
        data = parse_html_tables(
            """
                    <table border="1">
                        <thead>
                            <tr>
                                <th rowspan="2" colspan="2"></th>
                                <th colspan="3">Quarter 1</th>
                                <th colspan="3">Quarter 2</th>
                            </tr>
                            <tr>
                                <th>Jan</th>
                                <th>Feb</th>
                                <th>Mar</th>
                                <th>Apr</th>
                                <th>May</th>
                                <th>Jun</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <th rowspan="3">North</th>
                                <th>Sales</th>
                                <td>100</td>
                                <td>110</td>
                                <td>120</td>
                                <td>130</td>
                                <td>140</td>
                                <td>150</td>
                            </tr>
                            <tr>
                                <th>Cost</th>
                                <td>50</td>
                                <td>55</td>
                                <td>60</td>
                                <td>65</td>
                                <td>70</td>
                                <td>75</td>
                            </tr>
                            <tr>
                                <th>Profit</th>
                                <td>50</td>
                                <td>55</td>
                                <td>60</td>
                                <td>65</td>
                                <td>70</td>
                                <td>75</td>
                            </tr>
                            <tr>
                                <th rowspan="3">South</th>
                                <th>Sales</th>
                                <td>200</td>
                                <td>210</td>
                                <td>220</td>
                                <td>230</td>
                                <td>240</td>
                                <td>250</td>
                            </tr>
                            <tr>
                                <th>Cost</th>
                                <td>100</td>
                                <td>105</td>
                                <td>110</td>
                                <td>115</td>
                                <td>120</td>
                                <td>125</td>
                            </tr>
                            <tr>
                                <th>Profit</th>
                                <td>100</td>
                                <td>105</td>
                                <td>110</td>
                                <td>115</td>
                                <td>120</td>
                                <td>125</td>
                            </tr>
                        </tbody>
                    </table>"""
        )[0]

        print("\n=== Left Headers with Row Spans Test ===")
        print(data)

        self.assertTrue(data.is_rectangular)

        # Test top headers
        self.assertEqual(data.cell_text[0, 2], "Quarter 1")
        self.assertEqual(data.cell_text[0, 5], "Quarter 2")
        self.assertEqual(data.cell_text[1, 2], "Jan")
        self.assertEqual(data.cell_text[1, 3], "Feb")
        self.assertEqual(data.cell_text[1, 4], "Mar")

        # Test left headers with row spans
        self.assertEqual(data.cell_text[2, 0], "North")
        self.assertEqual(data.cell_text[2, 1], "Sales")
        self.assertEqual(data.cell_text[3, 1], "Cost")
        self.assertEqual(data.cell_text[4, 1], "Profit")

        self.assertEqual(data.cell_text[5, 0], "South")
        self.assertEqual(data.cell_text[5, 1], "Sales")

        # Test data values
        self.assertEqual(data.cell_text[2, 2], "100")
        self.assertEqual(data.cell_text[3, 2], "50")
        self.assertEqual(data.cell_text[5, 2], "200")

        # Test heading cells - should include both top headers and left headers
        # Top headers: rows 0-1, Left headers: column 0-1 in data rows
        self.assertIn((0, 2), data.heading_cells)  # Quarter 1
        self.assertIn((1, 2), data.heading_cells)  # Jan
        self.assertIn((2, 0), data.heading_cells)  # North (left header)
        self.assertIn((2, 1), data.heading_cells)  # Sales (left header)
        self.assertIn((5, 0), data.heading_cells)  # South (left header)

        # Test left heading relations with multiple levels
        # Data cell (2, 2) should have both North and Sales as left headers
        left_relations_2_2 = data.left_heading_relations(2, 2)
        self.assertIn((2, 0), left_relations_2_2)  # North
        self.assertIn((2, 1), left_relations_2_2)  # Sales

        # Data cell (3, 2) should have North (spans from row 2) and Cost as left headers
        left_relations_3_2 = data.left_heading_relations(3, 2)
        self.assertIn((2, 0), left_relations_3_2)  # North (row span)
        self.assertIn((3, 1), left_relations_3_2)  # Cost

        # Data cell (5, 3) should have South and Sales as left headers
        left_relations_5_3 = data.left_heading_relations(5, 3)
        self.assertIn((5, 0), left_relations_5_3)  # South
        self.assertIn((5, 1), left_relations_5_3)  # Sales

        # Test top heading relations
        top_relations_2_2 = data.top_heading_relations(2, 2)
        self.assertIn((0, 2), top_relations_2_2)  # Quarter 1
        self.assertIn((1, 2), top_relations_2_2)  # Jan

    def test_nested_header_groups_with_col_spans(self):
        """Test a complex table with nested header groups and various column spans"""
        data = parse_html_tables(
            """
                    <table border="1">
                        <thead>
                            <tr>
                                <th rowspan="4">Region</th>
                                <th rowspan="4">Store</th>
                                <th colspan="12">2024 Sales Data</th>
                            </tr>
                            <tr>
                                <th colspan="6">First Half</th>
                                <th colspan="6">Second Half</th>
                            </tr>
                            <tr>
                                <th colspan="3">Q1</th>
                                <th colspan="3">Q2</th>
                                <th colspan="3">Q3</th>
                                <th colspan="3">Q4</th>
                            </tr>
                            <tr>
                                <th>Jan</th>
                                <th>Feb</th>
                                <th>Mar</th>
                                <th>Apr</th>
                                <th>May</th>
                                <th>Jun</th>
                                <th>Jul</th>
                                <th>Aug</th>
                                <th>Sep</th>
                                <th>Oct</th>
                                <th>Nov</th>
                                <th>Dec</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td rowspan="2">West</td>
                                <td>Store A</td>
                                <td>10</td>
                                <td>11</td>
                                <td>12</td>
                                <td>13</td>
                                <td>14</td>
                                <td>15</td>
                                <td>16</td>
                                <td>17</td>
                                <td>18</td>
                                <td>19</td>
                                <td>20</td>
                                <td>21</td>
                            </tr>
                            <tr>
                                <td>Store B</td>
                                <td>22</td>
                                <td>23</td>
                                <td>24</td>
                                <td>25</td>
                                <td>26</td>
                                <td>27</td>
                                <td>28</td>
                                <td>29</td>
                                <td>30</td>
                                <td>31</td>
                                <td>32</td>
                                <td>33</td>
                            </tr>
                        </tbody>
                    </table>"""
        )[0]

        print("\n=== Nested Header Groups with Col Spans Test ===")
        print(data)

        self.assertTrue(data.is_rectangular)

        # Test nested header structure
        self.assertEqual(data.cell_text[0, 0], "Region")
        self.assertEqual(data.cell_text[0, 1], "Store")
        self.assertEqual(data.cell_text[0, 2], "2024 Sales Data")

        self.assertEqual(data.cell_text[1, 2], "First Half")
        self.assertEqual(data.cell_text[1, 8], "Second Half")

        self.assertEqual(data.cell_text[2, 2], "Q1")
        self.assertEqual(data.cell_text[2, 5], "Q2")
        self.assertEqual(data.cell_text[2, 8], "Q3")
        self.assertEqual(data.cell_text[2, 11], "Q4")

        self.assertEqual(data.cell_text[3, 2], "Jan")
        self.assertEqual(data.cell_text[3, 7], "Jun")
        self.assertEqual(data.cell_text[3, 13], "Dec")

        # Test data rows
        self.assertEqual(data.cell_text[4, 0], "West")
        self.assertEqual(data.cell_text[4, 1], "Store A")
        self.assertEqual(data.cell_text[4, 2], "10")
        self.assertEqual(data.cell_text[5, 1], "Store B")
        self.assertEqual(data.cell_text[5, 2], "22")

        # Test all header cells are marked
        self.assertIn((0, 0), data.heading_cells)  # Region
        self.assertIn((0, 1), data.heading_cells)  # Store
        self.assertIn((0, 2), data.heading_cells)  # 2024 Sales Data
        self.assertIn((1, 2), data.heading_cells)  # First Half
        self.assertIn((2, 2), data.heading_cells)  # Q1
        self.assertIn((3, 2), data.heading_cells)  # Jan

        # Test multiple top heading relations for a data cell
        # Cell (4, 2) - January data for Store A should have all 4 levels of headers
        top_relations_4_2 = data.top_heading_relations(4, 2)
        self.assertIn((0, 2), top_relations_4_2)  # 2024 Sales Data
        self.assertIn((1, 2), top_relations_4_2)  # First Half
        self.assertIn((2, 2), top_relations_4_2)  # Q1
        self.assertIn((3, 2), top_relations_4_2)  # Jan

        # Cell (4, 7) - June data should relate to Q2 and First Half
        top_relations_4_7 = data.top_heading_relations(4, 7)
        self.assertIn((0, 2), top_relations_4_7)  # 2024 Sales Data
        self.assertIn((1, 2), top_relations_4_7)  # First Half
        self.assertIn((2, 5), top_relations_4_7)  # Q2
        self.assertIn((3, 7), top_relations_4_7)  # Jun

        # Cell (4, 13) - December data should relate to Q4 and Second Half
        top_relations_4_13 = data.top_heading_relations(4, 13)
        self.assertIn((0, 2), top_relations_4_13)  # 2024 Sales Data
        self.assertIn((1, 8), top_relations_4_13)  # Second Half
        self.assertIn((2, 11), top_relations_4_13)  # Q4
        self.assertIn((3, 13), top_relations_4_13)  # Dec

        # Test left heading relations
        # Store B row that says "22" should relate to just West
        left_relations_5_2 = data.left_heading_relations(5, 2)
        self.assertEqual(data.cell_text[5, 2], "22")
        self.assertIn((4, 0), left_relations_5_2)
        self.assertEqual(len(left_relations_5_2), 1)

        # But of the left headings themselves at the top, January, should have both Region and Store
        self.assertEqual(data.left_heading_relations(3, 2), {(0, 0), (0, 1)})

    def test_large_real_table(self):
        html = """
<table border="1">
            <thead>
                <tr>
                    <th rowspan="2" class="left-align">Grupo Ocupacional</th>
                    <th rowspan="2" class="left-align">Classe Ocupacional</th>
                    <th colspan="2">Superior</th>
                    <th colspan="2">Médio</th>
                    <th colspan="2">Baixo</th>
                    <th colspan="2">Interior</th>
                    <th colspan="2">Ínfimo</th>
                    <th colspan="2">Total</th>
                </tr>
                <tr>
                    <th>N (1.000s)</th>
                    <th>%</th>
                    <th>N (1.000s)</th>
                    <th>%</th>
                    <th>N (1.000s)</th>
                    <th>%</th>
                    <th>N (1.000s)</th>
                    <th>%</th>
                    <th>N (1.000s)</th>
                    <th>%</th>
                    <th>N (1.000s)</th>
                    <th>%</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td rowspan="3" class="left-align">Empregadores</td>
                    <td class="left-align">A-1 Empregadores (> 10)</td>
                    <td>608</td>
                    <td>67,3</td>
                    <td>185</td>
                    <td>20,4</td>
                    <td>86</td>
                    <td>9,6</td>
                    <td>16</td>
                    <td>1,8</td>
                    <td>8</td>
                    <td>0,9</td>
                    <td>903</td>
                    <td>100</td>
                </tr>
                <tr>
                    <td class="left-align">A-2 Empregadores (<= 10)</td>
                    <td>1.555</td>
                    <td>36,9</td>
                    <td>1.107</td>
                    <td>26,3</td>
                    <td>1.036</td>
                    <td>24,7</td>
                    <td>341</td>
                    <td>8,1</td>
                    <td>171</td>
                    <td>4,1</td>
                    <td>4.213</td>
                    <td>100</td>
                </tr>
                <tr>
                    <td class="left-align">Total</td>
                    <td>2.162</td>
                    <td>42,3</td>
                    <td>1.292</td>
                    <td>25,3</td>
                    <td>1.126</td>
                    <td>22,0</td>
                    <td>357</td>
                    <td>7,0</td>
                    <td>179</td>
                    <td>3,5</td>
                    <td>5.116</td>
                    <td>100</td>
                </tr>
                <tr>
                    <td rowspan="3" class="left-align">Profissionais</td>
                    <td class="left-align">C Profissionais Autônomos</td>
                    <td>1.643</td>
                    <td>21,7</td>
                    <td>1.513</td>
                    <td>20,0</td>
                    <td>2.073</td>
                    <td>27,4</td>
                    <td>1.225</td>
                    <td>16,2</td>
                    <td>1.108</td>
                    <td>14,7</td>
                    <td>7.562</td>
                    <td>100</td>
                </tr>
                <tr>
                    <td class="left-align">D Profissionais Assalariados</td>
                    <td>4.438</td>
                    <td>13,3</td>
                    <td>6.030</td>
                    <td>18,0</td>
                    <td>11.550</td>
                    <td>34,5</td>
                    <td>7.027</td>
                    <td>21,0</td>
                    <td>4.389</td>
                    <td>13,1</td>
                    <td>33.434</td>
                    <td>100</td>
                </tr>
                <tr>
                    <td class="left-align">Total</td>
                    <td>6.081</td>
                    <td>14,8</td>
                    <td>7.543</td>
                    <td>18,4</td>
                    <td>13.623</td>
                    <td>33,2</td>
                    <td>8.252</td>
                    <td>20,1</td>
                    <td>5.497</td>
                    <td>13,4</td>
                    <td>40.995</td>
                    <td>100</td>
                </tr>
                <tr>
                    <td rowspan="4" class="left-align">Massa Não-Agrícola</td>
                    <td class="left-align">F Trabalhadores Autônomos</td>
                    <td>657</td>
                    <td>3,5</td>
                    <td>1.754</td>
                    <td>9,2</td>
                    <td>5.561</td>
                    <td>29,2</td>
                    <td>5.271</td>
                    <td>27,7</td>
                    <td>5.788</td>
                    <td>30,4</td>
                    <td>19.030</td>
                    <td>100</td>
                </tr>
                <tr>
                    <td class="left-align">G Trabalhadores Assalariados</td>
                    <td>282</td>
                    <td>0,7</td>
                    <td>1.657</td>
                    <td>4,3</td>
                    <td>10.363</td>
                    <td>27,1</td>
                    <td>13.002</td>
                    <td>34,0</td>
                    <td>12.968</td>
                    <td>33,9</td>
                    <td>38.272</td>
                    <td>100</td>
                </tr>
                <tr>
                    <td class="left-align">I Trabalhadores Domésticos</td>
                    <td>10</td>
                    <td>0,1</td>
                    <td>104</td>
                    <td>1,6</td>
                    <td>977</td>
                    <td>14,7</td>
                    <td>1.810</td>
                    <td>27,3</td>
                    <td>3.733</td>
                    <td>56,3</td>
                    <td>6.633</td>
                    <td>100</td>
                </tr>
                <tr>
                    <td class="left-align">Total</td>
                    <td>948</td>
                    <td>1,5</td>
                    <td>3.515</td>
                    <td>5,5</td>
                    <td>16.901</td>
                    <td>26,4</td>
                    <td>20.083</td>
                    <td>31,4</td>
                    <td>22.489</td>
                    <td>35,2</td>
                    <td>63.936</td>
                    <td>100</td>
                </tr>
                <tr>
                    <td rowspan="4" class="left-align">Massa Agrícola</td>
                    <td class="left-align">H-1 Proprietários Conta Própria</td>
                    <td>188</td>
                    <td>2,0</td>
                    <td>364</td>
                    <td>3,8</td>
                    <td>1.387</td>
                    <td>14,4</td>
                    <td>1.889</td>
                    <td>19,7</td>
                    <td>5.779</td>
                    <td>60,2</td>
                    <td>9.608</td>
                    <td>100</td>
                </tr>
                <tr>
                    <td class="left-align">H-2 Trabalhadores Autônomos</td>
                    <td>5</td>
                    <td>0,5</td>
                    <td>14</td>
                    <td>1,5</td>
                    <td>72</td>
                    <td>7,6</td>
                    <td>152</td>
                    <td>16,1</td>
                    <td>703</td>
                    <td>74,3</td>
                    <td>946</td>
                    <td>100</td>
                </tr>
                <tr>
                    <td class="left-align">H-3 Trabalhadores Assalariados</td>
                    <td>17</td>
                    <td>0,2</td>
                    <td>58</td>
                    <td>0,6</td>
                    <td>794</td>
                    <td>8,4</td>
                    <td>2.260</td>
                    <td>23,9</td>
                    <td>6.322</td>
                    <td>66,9</td>
                    <td>9.451</td>
                    <td>100</td>
                </tr>
                <tr>
                    <td class="left-align">Total</td>
                    <td>210</td>
                    <td>1,0</td>
                    <td>436</td>
                    <td>2,2</td>
                    <td>2.253</td>
                    <td>11,3</td>
                    <td>4.301</td>
                    <td>21,5</td>
                    <td>12.805</td>
                    <td>64,0</td>
                    <td>20.005</td>
                    <td>100</td>
                </tr>
                <tr>
                    <td rowspan="5" class="left-align">Não-remunerados</td>
                    <td class="left-align">Não-remunerados Não-Agrícolas</td>
                    <td>13</td>
                    <td>6,8</td>
                    <td>16</td>
                    <td>8,1</td>
                    <td>28</td>
                    <td>14,0</td>
                    <td>22</td>
                    <td>10,9</td>
                    <td>119</td>
                    <td>60,2</td>
                    <td>198</td>
                    <td>100</td>
                </tr>
                <tr>
                    <td class="left-align">Não-remunerados Agrícolas</td>
                    <td>5</td>
                    <td>0,1</td>
                    <td>13</td>
                    <td>0,3</td>
                    <td>59</td>
                    <td>1,6</td>
                    <td>352</td>
                    <td>9,4</td>
                    <td>3.302</td>
                    <td>88,5</td>
                    <td>3.731</td>
                    <td>100</td>
                </tr>
                <tr>
                    <td class="left-align">Sem Ocupação Com Renda</td>
                    <td>1.567</td>
                    <td>6,0</td>
                    <td>2.330</td>
                    <td>8,9</td>
                    <td>5.395</td>
                    <td>20,7</td>
                    <td>6.821</td>
                    <td>26,2</td>
                    <td>9.964</td>
                    <td>38,2</td>
                    <td>26.078</td>
                    <td>100</td>
                </tr>
                <tr>
                    <td class="left-align">Sem Ocupação Sem Renda</td>
                    <td></td>
                    <td></td>
                    <td></td>
                    <td></td>
                    <td></td>
                    <td></td>
                    <td></td>
                    <td></td>
                    <td>8.094</td>
                    <td>100</td>
                    <td>8.094</td>
                    <td>100</td>
                </tr>
                <tr>
                    <td class="left-align">Ignorados</td>
                    <td>177</td>
                    <td>10,3</td>
                    <td>202</td>
                    <td>11,8</td>
                    <td>364</td>
                    <td>21,1</td>
                    <td>337</td>
                    <td>19,6</td>
                    <td>640</td>
                    <td>37,2</td>
                    <td>1.720</td>
                    <td>100</td>
                </tr>
            </tbody>
        </table>
"""

        data = parse_html_tables(html)[0]

        self.assertTrue(data.is_rectangular)

    def test_non_rect(self):
        html = """<table>\n  <tr>\n    <th rowspan=\"2\">Model</th>\n    <th rowspan=\"2\">Avg FLOP×10<sup>23</sup></th>\n    <th colspan=\"5\">Dev Benchmarks</th>\n    <th colspan=\"4\">Held-out Evals</th>\n  </tr>\n  <tr>\n    <th>MMLU</th>\n    <th>ARC<sub>C</sub></th>\n    <th>HSwag</th>\n    <th>WinoG</th>\n    <th>NQ</th>\n    <th>DROP</th>\n    <th>AGIEval</th>\n    <th>GSM8K</th>\n    <th>MMLU<sub>PRO</sub></th>\n  </tr>\n  <tr>\n    <th colspan=\"11\">Open-weight models</th>\n  </tr>\n  <tr>\n    <td>Llama 2 13B</td>\n    <td>51.0</td>\n    <td>1.6</td>\n    <td>55.7</td>\n    <td>67.3</td>\n    <td>83.9</td>\n    <td>74.9</td>\n    <td>38.4</td>\n    <td>45.6</td>\n    <td>41.5</td>\n    <td>28.1</td>\n    <td>23.9</td>\n  </tr>\n  <tr>\n    <td>Mistral 7B</td>\n    <td>56.6</td>\n    <td>n/a</td>\n    <td>63.5</td>\n    <td>78.3</td>\n    <td>83.1</td>\n    <td>77.7</td>\n    <td>37.2</td>\n    <td>51.8</td>\n    <td>47.3</td>\n    <td>40.1</td>\n    <td>30.0</td>\n  </tr>\n  <tr>\n    <td>Llama 3.1 8B</td>\n    <td>59.7</td>\n    <td>7.2</td>\n    <td>66.9</td>\n    <td>79.5</td>\n    <td>81.6</td>\n    <td>76.6</td>\n    <td>33.9</td>\n    <td>56.4</td>\n    <td>51.3</td>\n    <td>56.5</td>\n    <td>34.7</td>\n  </tr>\n  <tr>\n    <td>Mistral Nemo 12B</td>\n    <td>64.9</td>\n    <td>n/a</td>\n    <td>69.5</td>\n    <td>85.2</td>\n    <td>85.6</td>\n    <td>81.5</td>\n    <td>39.7</td>\n    <td>69.2</td>\n    <td>54.7</td>\n    <td>62.1</td>\n    <td>36.7</td>\n  </tr>\n  <tr>\n    <td>Gemma 2 9B</td>\n    <td>66.3</td>\n    <td>4.4</td>\n    <td>70.6</td>\n    <td>89.5</td>\n    <td>87.3</td>\n    <td>78.8</td>\n    <td>38.0</td>\n    <td>63.0</td>\n    <td>57.3</td>\n    <td>70.1</td>\n    <td>42.0</td>\n  </tr>\n  <tr>\n    <td>Qwen 2.5 7B</td>\n    <td>67.2</td>\n    <td>8.2</td>\n    <td>74.4</td>\n    <td>89.5</td>\n    <td>89.7</td>\n    <td>74.2</td>\n    <td>29.9</td>\n    <td>55.8</td>\n    <td>63.7</td>\n    <td>81.5</td>\n    <td>45.8</td>\n  </tr>\n  <tr>\n    <td>Qwen 2.5 14B</td>\n    <td>71.5</td>\n    <td>16.0</td>\n    <td>79.3</td>\n    <td>94.0</td>\n    <td>94.0</td>\n    <td>80.0</td>\n    <td>37.3</td>\n    <td>51.5</td>\n    <td>71.0</td>\n    <td>83.4</td>\n    <td>52.8</td>\n  </tr>\n  <tr>\n    <th colspan=\"11\">Models with partially available data</th>\n  </tr>\n  <tr>\n    <td>StableLM 2 12B</td>\n    <td>60.2</td>\n    <td>2.9</td>\n    <td>62.4</td>\n    <td>81.9</td>\n    <td>84.5</td>\n    <td>77.7</td>\n    <td>37.6</td>\n    <td>55.5</td>\n    <td>50.9</td>\n    <td>62.0</td>\n    <td>29.3</td>\n  </tr>\n  <tr>\n    <td>Zamba 2 7B</td>\n    <td>63.7</td>\n    <td>n/c</td>\n    <td>68.5</td>\n    <td>92.2</td>\n    <td>89.4</td>\n    <td>79.6</td>\n    <td>36.5</td>\n    <td>51.7</td>\n    <td>55.5</td>\n    <td>67.2</td>\n    <td>32.8</td>\n  </tr>\n  <tr>\n    <th colspan=\"11\">Fully-open models</th>\n  </tr>\n  <tr>\n    <td>Amber 7B</td>\n    <td>32.5</td>\n    <td>0.5</td>\n    <td>24.7</td>\n    <td>44.9</td>\n    <td>74.5</td>\n    <td>65.5</td>\n    <td>18.7</td>\n    <td>26.1</td>\n    <td>21.8</td>\n    <td>4.8</td>\n    <td>11.7</td>\n  </tr>\n  <tr>\n    <td>OLMo 7B</td>\n    <td>35.4</td>\n    <td>1.0</td>\n    <td>28.3</td>\n    <td>46.4</td>\n    <td>78.1</td>\n    <td>68.5</td>\n    <td>24.8</td>\n    <td>27.3</td>\n    <td>23.7</td>\n    <td>9.2</td>\n    <td>12.1</td>\n  </tr>\n  <tr>\n    <td>MAP Neo 7B</td>\n    <td>47.9</td>\n    <td>2.1</td>\n    <td>58.0</td>\n    <td>78.4</td>\n    <td>72.8</td>\n    <td>69.2</td>\n    <td>28.9</td>\n    <td>39.4</td>\n    <td>45.8</td>\n    <td>12.5</td>\n    <td>25.9</td>\n  </tr>\n  <tr>\n    <td>OLMo 0424 7B</td>\n    <td>49.8</td>\n    <td>1.0</td>\n    <td>54.3</td>\n    <td>66.9</td>\n    <td>80.1</td>\n    <td>73.6</td>\n    <td>29.6</td>\n    <td>50.0</td>\n    <td>43.9</td>\n    <td>27.7</td>\n    <td>22.1</td>\n  </tr>\n  <tr>\n    <td>DCLM 7B</td>\n    <td>55.2</td>\n    <td>1.0</td>\n    <td>64.4</td>\n    <td>79.8</td>\n    <td>82.3</td>\n    <td>77.3</td>\n    <td>28.8</td>\n    <td>39.3</td>\n    <td>47.5</td>\n    <td>46.1</td>\n    <td>31.3</td>\n  </tr>\n  <tr>\n    <td>OLMo 2 7B</td>\n    <td>61.2</td>\n    <td>1.8</td>\n    <td>63.7</td>\n    <td>79.8</td>\n    <td>83.8</td>\n    <td>77.2</td>\n    <td>36.9</td>\n    <td>60.8</td>\n    <td>50.4</td>\n    <td>67.5</td>\n    <td>31.0</td>\n  </tr>\n  <tr>\n    <td>OLMo 2 13B</td>\n    <td>66.8</td>\n    <td>4.6</td>\n    <td>67.5</td>\n    <td>83.5</td>\n    <td>86.4</td>\n    <td>81.5</td>\n    <td>46.7</td>\n    <td>70.7</td>\n    <td>54.2</td>\n    <td>75.1</td>\n    <td>35.1</td>\n  </tr>\n</table>"""

        data = parse_html_tables(html)[0]

        self.assertFalse(data.is_rectangular)