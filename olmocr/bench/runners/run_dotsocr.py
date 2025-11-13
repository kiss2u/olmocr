import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

from pypdf import PdfReader, PdfWriter

REPO_ROOT = Path(__file__).resolve().parents[3]
DOTS_OCR_PATH = REPO_ROOT / "dots.ocr"

if not DOTS_OCR_PATH.exists():
    raise ImportError(f"Could not find dots.ocr checkout at {DOTS_OCR_PATH}")

if str(DOTS_OCR_PATH) not in sys.path:
    sys.path.insert(0, str(DOTS_OCR_PATH))

from dots_ocr.parser import DotsOCRParser  # noqa: E402

_PARSER_CACHE: Dict[
    Tuple[str, str, int, str, float, float, int, int, Optional[int], Optional[int], bool],
    DotsOCRParser,
] = {}


def _parse_server(server: Optional[str]) -> Tuple[str, str, int]:
    normalized = (server or "").strip() or "http://localhost:8000"
    if "://" not in normalized:
        normalized = f"http://{normalized}"

    parsed = urlparse(normalized)
    if not parsed.scheme or not parsed.hostname:
        raise ValueError(f"Invalid server URL: {server}")

    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80

    return parsed.scheme, parsed.hostname, port


def _get_parser(
    *,
    protocol: str,
    ip: str,
    port: int,
    model_name: str,
    temperature: float,
    top_p: float,
    max_completion_tokens: int,
    num_thread: int,
    dpi: int,
    min_pixels: Optional[int],
    max_pixels: Optional[int],
    use_hf: bool,
) -> DotsOCRParser:
    cache_key = (
        protocol,
        ip,
        port,
        model_name,
        temperature,
        top_p,
        max_completion_tokens,
        num_thread,
        dpi,
        min_pixels,
        max_pixels,
        use_hf,
    )

    parser = _PARSER_CACHE.get(cache_key)
    if parser is None:
        parser = DotsOCRParser(
            protocol=protocol,
            ip=ip,
            port=port,
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            max_completion_tokens=max_completion_tokens,
            num_thread=num_thread,
            dpi=dpi,
            output_dir=tempfile.mkdtemp(prefix="dotsocr-cache-"),
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            use_hf=use_hf,
        )
        _PARSER_CACHE[cache_key] = parser

    return parser


def _extract_markdown_from_pdf(
    parser: DotsOCRParser,
    pdf_path: str,
    page_num: int,
    prompt_mode: str,
    fitz_preprocess: bool,
) -> str:
    if page_num < 1:
        raise ValueError("page_num must be >= 1")

    with open(pdf_path, "rb") as pdf_file:
        reader = PdfReader(pdf_file)
        zero_index = page_num - 1
        if zero_index >= len(reader.pages):
            raise ValueError(f"Page {page_num} does not exist in {pdf_path}")

        writer = PdfWriter()
        writer.add_page(reader.pages[zero_index])

        with tempfile.TemporaryDirectory(prefix="dotsocr-") as tmp_dir:
            single_page_pdf = os.path.join(tmp_dir, "page.pdf")
            with open(single_page_pdf, "wb") as tmp_pdf:
                writer.write(tmp_pdf)

            output_dir = os.path.join(tmp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)

            results = parser.parse_file(
                single_page_pdf,
                output_dir=output_dir,
                prompt_mode=prompt_mode,
                fitz_preprocess=fitz_preprocess,
            )

            if not results:
                raise RuntimeError("DotsOCR did not return any results")

            # Single-page PDF means the first result corresponds to our request.
            page_result = results[0]
            md_path = page_result.get("md_content_path") or page_result.get("md_content_nohf_path")

            if not md_path:
                raise RuntimeError(
                    f"DotsOCR (prompt_mode={prompt_mode}) did not produce markdown output for page {page_num}"
                )

            if not os.path.exists(md_path):
                raise RuntimeError(f"DotsOCR reported markdown at '{md_path}', but the file does not exist")

            with open(md_path, "r", encoding="utf-8") as md_file:
                return md_file.read()


async def run_dotsocr(
    pdf_path: str,
    page_num: int = 1,
    server: Optional[str] = "http://localhost:8000",
    prompt_mode: str = "prompt_layout_all_en",
    model_name: str = "rednote-hilab/dots.ocr",
    temperature: float = 0.1,
    top_p: float = 1.0,
    max_completion_tokens: int = 16384,
    num_thread: int = 16,
    dpi: int = 200,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    use_hf: bool = False,
    fitz_preprocess: bool = False,
) -> str:
    """
    Run DotsOCR on a single PDF page and return the markdown output.
    """

    protocol, ip, port = _parse_server(server)
    parser = _get_parser(
        protocol=protocol,
        ip=ip,
        port=port,
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        max_completion_tokens=max_completion_tokens,
        num_thread=num_thread,
        dpi=dpi,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        use_hf=use_hf,
    )

    return await asyncio.to_thread(
        _extract_markdown_from_pdf,
        parser,
        pdf_path,
        page_num,
        prompt_mode,
        fitz_preprocess,
    )
