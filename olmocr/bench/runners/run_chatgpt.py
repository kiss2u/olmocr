import json
import os
from typing import Literal

from openai import OpenAI

from olmocr.bench.prompts import (
    build_basic_prompt,
    build_openai_silver_data_prompt_no_document_anchoring,
)
from olmocr.data.renderpdf import (
    get_png_dimensions_from_base64,
    render_pdf_to_base64png,
)
from olmocr.prompts.anchor import get_anchor_text
from olmocr.prompts.prompts import (
    PageResponse,
    build_finetuning_prompt,
    build_openai_silver_data_prompt,
    build_openai_silver_data_prompt_v2,
    build_openai_silver_data_prompt_v2_simple,
    build_openai_silver_data_prompt_v3_simple,
    openai_response_format_schema,
)

# Global variables to track token usage and document count
TOTAL_INPUT_TOKENS = 0
TOTAL_OUTPUT_TOKENS = 0
TOTAL_DOCUMENTS = 0


def run_chatgpt(
    pdf_path: str,
    page_num: int = 1,
    model: str = "gpt-4o-2024-08-06",
    temperature: float = 0.1,
    target_longest_image_dim: int = 2048,
    max_completion_tokens: int=10000,
    prompt_template: Literal["full", "full_no_document_anchoring", "basic", "finetune", "fullv2", "fullv2simple", "fullv3simple"] = "finetune",
    response_template: Literal["plain", "json"] = "json",
) -> str:
    """
    Convert page of a PDF file to markdown using the commercial openAI APIs.

    See run_server.py for running against an openai compatible server

    Args:
        pdf_path (str): The local path to the PDF file.

    Returns:
        str: The OCR result in markdown format.
    """
    global TOTAL_INPUT_TOKENS, TOTAL_OUTPUT_TOKENS, TOTAL_DOCUMENTS
    # Convert the first page of the PDF to a base64-encoded PNG image.
    image_base64 = render_pdf_to_base64png(pdf_path, page_num=page_num, target_longest_image_dim=target_longest_image_dim)
    anchor_text = get_anchor_text(pdf_path, page_num, pdf_engine="pdfreport")

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("You must specify an OPENAI_API_KEY")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if prompt_template == "full":
        prompt = build_openai_silver_data_prompt(anchor_text)
    elif prompt_template == "full_no_document_anchoring":
        prompt = build_openai_silver_data_prompt_no_document_anchoring(anchor_text)
    elif prompt_template == "finetune":
        prompt = build_finetuning_prompt(anchor_text)
    elif prompt_template == "basic":
        prompt = build_basic_prompt()
    elif prompt_template == "fullv2":
        prompt = build_openai_silver_data_prompt_v2(anchor_text)
    elif prompt_template == "fullv2simple":
        width, height = get_png_dimensions_from_base64(image_base64)
        prompt = build_openai_silver_data_prompt_v2_simple(width, height)
    elif prompt_template == "fullv3simple":
        width, height = get_png_dimensions_from_base64(image_base64)
        prompt = build_openai_silver_data_prompt_v3_simple(width, height)
    else:
        raise ValueError("Unknown prompt template")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            }
        ],
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        # reasoning_effort="high",
        response_format=openai_response_format_schema() if response_template == "json" else None,
        safety_identifier="olmocr-bench-runner",
    )

    # Accumulate token counts from the response
    if response.usage:
        TOTAL_INPUT_TOKENS += response.usage.prompt_tokens
        TOTAL_OUTPUT_TOKENS += response.usage.completion_tokens

    # Increment document counter
    TOTAL_DOCUMENTS += 1

    raw_response = response.choices[0].message.content

    assert len(response.choices) > 0
    assert response.choices[0].message.refusal is None
    assert response.choices[0].finish_reason == "stop"

    if response_template == "json":
        data = json.loads(raw_response)
        data = PageResponse(**data)

        # Print token counts before returning
        print(f"Token Usage - Documents: {TOTAL_DOCUMENTS}, Input: {TOTAL_INPUT_TOKENS}, Output: {TOTAL_OUTPUT_TOKENS}")
        return data.natural_text
    else:
        # Print token counts before returning
        print(f"Token Usage - Documents: {TOTAL_DOCUMENTS}, Input: {TOTAL_INPUT_TOKENS}, Output: {TOTAL_OUTPUT_TOKENS}")
        return raw_response
