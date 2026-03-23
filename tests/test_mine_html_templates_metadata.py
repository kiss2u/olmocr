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


class TestExtractHtmlMetadata(unittest.TestCase):
    def test_extract_metadata_portuguese_document(self):
        """Test metadata extraction from a Portuguese document with mixed content."""
        html_content = """
        <html lang="pt">
        <head><title>Test Document</title></head>
        <body>
            <header>Header content here</header>
            <h1>Política de Metadados</h1>
            <p>Este é um documento de teste com texto em português.</p>
            <p>Contém múltiplos parágrafos para simular conteúdo real.</p>
            <div class="image">Image placeholder 1</div>
            <p>Mais texto após a imagem.</p>
            <footer>Footer content</footer>
        </body>
        </html>
        """

        metadata = extract_html_metadata(html_content)

        # Check language extraction
        self.assertEqual(metadata["primary_language"], "pt")

        # Check rotation values (always fixed)
        self.assertTrue(metadata["is_rotation_valid"])
        self.assertEqual(metadata["rotation_correction"], 0)

        # Check table/diagram detection
        # With 1 image (500 chars) and small text content, image ratio > 50%
        self.assertFalse(metadata["is_table"])
        self.assertTrue(metadata["is_diagram"])  # Image estimate dominates

    def test_extract_metadata_table_heavy_document(self):
        """Test metadata extraction from a document that is mostly tables."""
        html_content = """
        <html lang="en">
        <body>
            <p>Small intro text</p>
            <table>
                <tr><td>Cell 1</td><td>Cell 2</td><td>Cell 3</td></tr>
                <tr><td>Data A</td><td>Data B</td><td>Data C</td></tr>
                <tr><td>More data</td><td>More data</td><td>More data</td></tr>
                <tr><td>Even more data</td><td>Even more data</td><td>Even more data</td></tr>
                <tr><td>Lots of data</td><td>Lots of data</td><td>Lots of data</td></tr>
                <tr><td>Table content</td><td>Table content</td><td>Table content</td></tr>
                <tr><td>Final row</td><td>Final row</td><td>Final row</td></tr>
            </table>
        </body>
        </html>
        """

        metadata = extract_html_metadata(html_content)

        self.assertEqual(metadata["primary_language"], "en")
        self.assertTrue(metadata["is_table"])  # Should be True as >50% is table
        self.assertFalse(metadata["is_diagram"])

    def test_extract_metadata_image_heavy_document(self):
        """Test metadata extraction from a document that is mostly images."""
        html_content = """
        <html lang="es">
        <body>
            <p>Brief text</p>
            <div class="image">Image 1</div>
            <div class="image">Image 2</div>
            <div class="image">Image 3</div>
            <div class="image">Image 4</div>
            <div class="image">Image 5</div>
        </body>
        </html>
        """

        metadata = extract_html_metadata(html_content)

        self.assertEqual(metadata["primary_language"], "es")
        self.assertFalse(metadata["is_table"])
        self.assertTrue(metadata["is_diagram"])  # Should be True as >50% is images

    def test_extract_metadata_language_with_region(self):
        """Test that language codes with regions (e.g., pt-BR) are shortened."""
        html_content = """
        <html lang="pt-BR">
        <body>
            <p>Texto em português brasileiro</p>
        </body>
        </html>
        """

        metadata = extract_html_metadata(html_content)

        # Should convert pt-BR to pt
        self.assertEqual(metadata["primary_language"], "pt")

    def test_extract_metadata_no_html_tag(self):
        """Test extraction when there's no html tag (defaults to 'en')."""
        html_content = """
        <body>
            <p>Content without html tag</p>
        </body>
        """

        metadata = extract_html_metadata(html_content)

        self.assertEqual(metadata["primary_language"], "en")  # Should default to 'en'

    def test_extract_metadata_mixed_content(self):
        """Test a document with mixed content types."""
        html_content = """<!DOCTYPE html>
                <html lang="pt-BR">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Política de Metadados para Livros e Capítulos de Livro UFPA</title>
                    <style>
                        body {
                            font-family: Arial, sans-serif;
                            margin: 0;
                            padding: 20px;
                            width: 725px;
                            height: 1024px;
                            box-sizing: border-box;
                        }
                        
                        .header-logos {
                            display: flex;
                            justify-content: space-between;
                            align-items: center;
                            margin-bottom: 30px;
                        }
                        
                        .image {
                            border: 2px solid black;
                            background-color: #ccc;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            color: black;
                            font-weight: bold;
                        }
                        
                        .logo-left {
                            width: 120px;
                            height: 80px;
                        }
                        
                        .logo-center {
                            width: 300px;
                            height: 80px;
                        }
                        
                        .logo-right {
                            width: 120px;
                            height: 80px;
                        }
                        
                        h1 {
                            text-align: center;
                            font-weight: bold;
                            font-size: 16px;
                            margin: 20px 0;
                            text-transform: uppercase;
                        }
                        
                        .intro-text {
                            text-align: justify;
                            margin-bottom: 20px;
                            font-size: 14px;
                            line-height: 1.4;
                        }
                        
                        table {
                            width: 100%;
                            border-collapse: collapse;
                            font-size: 12px;
                        }
                        
                        th, td {
                            border: 1px solid black;
                            padding: 8px;
                            text-align: left;
                            vertical-align: middle;
                        }
                        
                        th {
                            background-color: #f0f0f0;
                            font-weight: bold;
                            text-align: center;
                        }
                        
                        .col-metadados {
                            width: 35%;
                        }
                        
                        .col-valor {
                            width: 35%;
                        }
                        
                        .col-repetitivo {
                            width: 15%;
                            text-align: center;
                        }
                        
                        .col-condicao {
                            width: 15%;
                            text-align: center;
                        }
                        
                        footer {
                            text-align: center;
                            margin-top: 20px;
                            font-size: 14px;
                            font-weight: bold;
                        }
                    </style>
                </head>
                <body>
                    <header>
                        <div class="header-logos">
                            <div class="image logo-left">Biblioteca Central UFPA</div>
                            <div class="image logo-center">LIVRO ABERTO portal do livro aberto da UFPA</div>
                            <div class="image logo-right">SIBI/UFPA</div>
                        </div>
                    </header>

                    <main>
                        <h1>Política de Metadados para Livros e Capítulos de Livro UFPA</h1>
                        
                        <p class="intro-text">
                            Essa política de metadados possui o objetivo de garantir a consistência do trabalho executado no Portal do Livro Aberto. Dessa forma, foi desenvolvido com base no esquema de metadados do Dublin Core com adaptações para a realidade brasileira e local.
                        </p>

                        <table>
                            <thead>
                                <tr>
                                    <th class="col-metadados">METADADOS</th>
                                    <th class="col-valor">VALOR</th>
                                    <th class="col-repetitivo">REPETITIVO</th>
                                    <th class="col-condicao">CONDIÇÃO</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>dc.type</td>
                                    <td>Tipo de documento</td>
                                    <td>Não</td>
                                    <td>Obrigatório</td>
                                </tr>
                                <tr>
                                    <td>dc.title</td>
                                    <td>Título e subtítulo (se houver)</td>
                                    <td>Não</td>
                                    <td>Obrigatório</td>
                                </tr>
                                <tr>
                                    <td>dc.title.alternative</td>
                                    <td>Título alternativo</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.creator</td>
                                    <td>Autor</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.creator.Lattes</td>
                                    <td>URL do currículo Lattes do autor</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.creator.ORCID</td>
                                    <td>ORCID do autor</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.description.affiliation</td>
                                    <td>Afiliação do autor</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.contributor.organizer</td>
                                    <td>Organizador</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.contributor.organizerLattes</td>
                                    <td>URL do currículo Lattes do organizador</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.contributor.organizerORCID</td>
                                    <td>ORCID do organizador</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.description.affiliationOrganizer</td>
                                    <td>Afiliação do organizador</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.contributor.coordinator</td>
                                    <td>Coordenador</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.contributor.coordinatorLattes</td>
                                    <td>URL do currículo Lattes do coordenador</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.contributor.coordinatorORCID</td>
                                    <td>ORCID do coordenador</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.contributor.affiliationCoordinator</td>
                                    <td>Afiliação do coordenador</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.contributor.editor</td>
                                    <td>Editor</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.contributor.editorLattes</td>
                                    <td>URL do currículo Lattes do editor</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.contributor.editorORCID</td>
                                    <td>ORCID do editor</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                                <tr>
                                    <td>dc.description.affiliationEditor</td>
                                    <td>Afiliação do editor</td>
                                    <td>Sim</td>
                                    <td>Opcional</td>
                                </tr>
                            </tbody>
                        </table>
                    </main>

                    <footer>
                        <div>3</div>
                    </footer>
                </body>
                </html>
        """

        metadata = extract_html_metadata(html_content)

        self.assertEqual(metadata["primary_language"], "pt")
        self.assertTrue(metadata["is_table"])
        self.assertFalse(metadata["is_diagram"])

    def test_extract_metadata_empty_body(self):
        """Test extraction with empty or minimal content."""
        html_content = """
        <html lang="de">
        <body></body>
        </html>
        """

        metadata = extract_html_metadata(html_content)

        self.assertEqual(metadata["primary_language"], "de")
        self.assertFalse(metadata["is_table"])
        self.assertFalse(metadata["is_diagram"])
        self.assertTrue(metadata["is_rotation_valid"])
        self.assertEqual(metadata["rotation_correction"], 0)


if __name__ == "__main__":
    unittest.main()
