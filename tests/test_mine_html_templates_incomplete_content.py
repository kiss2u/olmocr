import asyncio
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from bs4 import BeautifulSoup

from olmocr.bench.tests import TestType
from olmocr.synth.mine_html_templates import render_pdf_with_playwright


class TestIncompleteContent(unittest.TestCase):
    def test_pg39(self):
        html = """<!DOCTYPE html>

<html lang="en">
<head><meta content="d1fdb60869ea7e1e0c48228f80477b67661e7654" name="olmocr_git_commit"/>
<meta charset="utf-8"/>
<meta content="width=768, height=1024" name="viewport"/>
<title>History of Computing</title>
<style>
        body {
            margin: 0;
            padding: 0;
            font-family: Georgia, serif;
            background: white;
        }
        .page {
            width: 768px;
            height: 1024px;
            position: relative;
            margin: 0 auto;
            background: white;
            box-sizing: border-box;
            page-break-after: always;
        }
        .page-border {
            position: absolute;
            top: 20px;
            left: 20px;
            right: 20px;
            bottom: 20px;
            border: 3px solid black;
        }
        .decorative-stripes {
            position: absolute;
            left: 30px;
            top: 30px;
            bottom: 30px;
            width: 80px;
        }
        .stripe {
            position: absolute;
            top: 0;
            bottom: 0;
            opacity: 0.3;
        }
        .stripe1 { left: 0; width: 20px; background: #ffcccc; }
        .stripe2 { left: 25px; width: 15px; background: #ffb3b3; }
        .stripe3 { left: 45px; width: 25px; background: #ffd9cc; }
        .stripe4 { left: 75px; width: 15px; background: #fff0e6; }
        
        .circle {
            position: absolute;
            border-radius: 50%;
            background: #ff8833;
        }
        .circle1 { width: 130px; height: 130px; left: 145px; top: 260px; }
        .circle2 { width: 45px; height: 45px; left: 235px; top: 335px; }
        .circle3 { width: 70px; height: 70px; left: 205px; top: 360px; }
        .circle4 { width: 25px; height: 25px; left: 188px; top: 395px; }
        .circle5 { width: 18px; height: 18px; left: 225px; top: 415px; }
        
        header.date {
            position: absolute;
            top: 5px;
            right: 30px;
            font-family: Arial, sans-serif;
            font-size: 14px;
        }
        
        .title-content {
            position: absolute;
            left: 260px;
            top: 340px;
        }
        
        h1 {
            font-size: 30px;
            color: #666;
            margin: 0;
            letter-spacing: 3px;
            font-weight: normal;
            font-variant: small-caps;
        }
        
        .subtitle {
            font-size: 16px;
            color: #333;
            margin: 8px 0 5px 0;
            font-family: Georgia, serif;
        }
        
        .date-line {
            font-size: 16px;
            color: #333;
            margin: 0;
            font-family: Georgia, serif;
        }
        
        /* Page 2 styles */
        .content-page {
            padding: 70px 60px 60px 120px;
        }
        
        h2 {
            font-size: 26px;
            color: #888;
            margin: 0 0 40px 0;
            font-weight: normal;
            letter-spacing: 1px;
            text-transform: uppercase;
        }
        
        .timeline-item {
            margin-bottom: 35px;
            position: relative;
            padding-left: 25px;
        }
        
        .timeline-item::before {
            content: '';
            position: absolute;
            left: 0;
            top: 7px;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #ff8833;
        }
        
        .timeline-text {
            font-size: 20px;
            line-height: 1.4;
            color: #333;
            margin: 0;
        }
        
        .abacus-image {
            position: absolute;
            right: 130px;
            top: 150px;
            width: 180px;
            height: 110px;
            background: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="180" height="110"><rect fill="%23654321" width="180" height="110" rx="5"/><rect fill="%23333" x="10" y="30" width="160" height="2"/><rect fill="%23333" x="10" y="50" width="160" height="2"/><rect fill="%23333" x="10" y="70" width="160" height="2"/><rect fill="%23333" x="10" y="90" width="160" height="2"/></svg>');
            background-size: cover;
            border: 2px solid #333;
        }
        
        .abacus-labels {
            position: absolute;
            right: 70px;
            top: 150px;
            font-size: 11px;
            font-family: Arial, sans-serif;
        }
        
        .label {
            margin: 4px 0;
            white-space: nowrap;
        }
        
        .highlight-box {
            position: absolute;
            right: 90px;
            bottom: 200px;
            width: 240px;
            background: #ffff66;
            padding: 18px;
            font-family: Arial, sans-serif;
            font-size: 17px;
            line-height: 1.5;
            border: 1px solid #e6e600;
        }
        
        .highlight-marker {
            position: absolute;
            left: -15px;
            top: 50%;
            transform: translateY(-50%);
            width: 12px;
            height: 12px;
            background: #ffff00;
            border-radius: 50%;
        }
        
        .circle-page2 {
            width: 85px;
            height: 85px;
            position: absolute;
            right: 55px;
            bottom: 85px;
            background: #ff8833;
            border-radius: 50%;
        }
        
        footer {
            position: absolute;
            bottom: 30px;
            right: 40px;
            font-family: Arial, sans-serif;
            font-size: 14px;
            color: #333;
        }
    </style>
</head>
<body>
<!-- Page 1: Title Slide -->
<div class="page">
<header class="date">9/2/2010</header>
<div class="page-border"></div>
<div class="decorative-stripes">
<div class="stripe stripe1"></div>
<div class="stripe stripe2"></div>
<div class="stripe stripe3"></div>
<div class="stripe stripe4"></div>
</div>
<div class="circle circle1"></div>
<div class="circle circle2"></div>
<div class="circle circle3"></div>
<div class="circle circle4"></div>
<div class="circle circle5"></div>
<div class="title-content">
<h1>HISTORY OF COMPUTING</h1>
<p class="subtitle">Fall 2010 - CSE 111</p>
<p class="date-line">Friday, September 3 &amp; Wednesday, September 8</p>
</div>
</div>
<!-- Page 2: Content Slide -->
<div class="page">
<div class="page-border"></div>
<div class="decorative-stripes">
<div class="stripe stripe1"></div>
<div class="stripe stripe2"></div>
<div class="stripe stripe3"></div>
<div class="stripe stripe4"></div>
</div>
<div class="content-page">
<h2>PRE 1642</h2>
<div class="timeline-item">
<p class="timeline-text">3000 BC – Abacus is invented in Babylonia</p>
</div>
<div class="timeline-item">
<p class="timeline-text">800 AD - Chinese begin to use zero.</p>
</div>
<div class="abacus-image"></div>
<div class="abacus-labels">
<div class="label">← Exo</div>
<div class="label" style="margin-top: 12px;">← Span</div>
<div class="label" style="margin-top: 12px;">← Rods</div>
<div class="label" style="margin-top: 12px;">← Frame</div>
</div>
<div class="highlight-box">
<div class="highlight-marker"></div>
                From 800 to 1641 some kinda important stuff happens in math – algebra, calculus, geometry, etc.
            </div>
</div>
<div class="circle-page2"></div>
<footer>1</footer>
</div>
</body>
</html>"""

        with tempfile.TemporaryDirectory() as td:
            rendered = asyncio.run(render_pdf_with_playwright(html, os.path.join(td, "temp.pdf"), 768, 1024))

        # Somehow we need to error out and show that this page didn't render fully
        self.assertFalse(rendered)


if __name__ == "__main__":
    unittest.main()
