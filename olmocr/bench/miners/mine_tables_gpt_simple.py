#!/usr/bin/env python3
"""
mine_tables_gpt_simple.py - Identify PDF documents with tables and copy them.

This script:
1. Takes a file containing S3 paths to PDF documents as input
2. For each PDF, renders a random page and uses GPT-4o to check for tables
3. Identifies PDFs where the page contains a table
4. Copies those PDF files to a new output folder

Usage:
  python mine_tables_gpt_simple.py --input_list path/to/s3_paths.txt --output_dir path/to/output --api_key your_openai_api_key
"""

import argparse
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import boto3
import pypdf
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm

from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.filter import PdfFilter

TARGET_IMAGE_DIM = 1024


class TableInfo(BaseModel):
    """Information about a single table."""

    num_rows: int
    num_cols: int


class TableDetectionResponse(BaseModel):
    """Structured output for table detection."""

    tables: list[TableInfo]


def download_pdf_from_s3(s3_path: str, local_path: str) -> bool:
    """
    Download a PDF file from S3.

    Args:
        s3_path: The S3 path (s3://bucket/path/to/file.pdf)
        local_path: The local path to save the file

    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        # Parse S3 path
        parts = s3_path.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        key = parts[1]

        # Create S3 client
        s3 = boto3.client("s3")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Download file
        s3.download_file(bucket, key, local_path)
        return True
    except Exception as e:
        print(f"Error downloading {s3_path}: {str(e)}")
        return False


def get_cell_count_bucket(total_cells: int) -> str:
    """
    Get the folder name for a given cell count, bucketed by powers of 2.

    Args:
        total_cells: Total number of cells across all tables

    Returns:
        str: Folder name like "0_cells", "1_cell", "2_cells", "4_cells", etc.
    """
    if total_cells == 0:
        return "0_cells"
    elif total_cells == 1:
        return "1_cell"
    else:
        # Find the next power of 2 >= total_cells
        power = 1
        while power < total_cells:
            power *= 2
        return f"{power}_cells"


def check_for_table(pdf_path: str, page_num: int, api_key: str) -> Optional[tuple[bool, int]]:
    """
    Use GPT-4o to check if a page contains a table.

    Args:
        pdf_path: Path to the PDF file
        page_num: The page number to analyze (0-indexed)
        api_key: OpenAI API key

    Returns:
        Optional[tuple[bool, int]]: Tuple of (has_table, total_cells) or None if detection fails
    """
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    try:
        # Render the PDF page as an image (render_pdf_to_base64png is 1-indexed)
        image_base64 = render_pdf_to_base64png(pdf_path, page_num=page_num + 1, target_longest_image_dim=TARGET_IMAGE_DIM)

        # Prompt asking for detailed table information
        prompt = "Identify all tables on this page. For each table, count the number of rows and columns. Return an empty list if there are no tables."

        response = client.beta.chat.completions.parse(
            model="gpt-5.1",
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}],
                }
            ],
            max_completion_tokens=1000,
            response_format=TableDetectionResponse,
        )

        if not response.choices or len(response.choices) == 0:
            print(f"No response generated for {pdf_path} page {page_num}")
            return None

        # Parse the structured response
        parsed_response = response.choices[0].message.parsed

        if parsed_response is None:
            print(f"Failed to parse response for {pdf_path} page {page_num}")
            return None

        tables = parsed_response.tables
        has_table = len(tables) > 0
        total_cells = sum(table.num_rows * table.num_cols for table in tables)

        if has_table:
            print(f"Found {len(tables)} table(s) in {pdf_path} page {page_num + 1}, total cells: {total_cells}")
            for i, table in enumerate(tables, 1):
                print(f"  Table {i}: {table.num_rows} rows × {table.num_cols} cols = {table.num_rows * table.num_cols} cells")

        return (has_table, total_cells)

    except Exception as e:
        print(f"Error checking {pdf_path} page {page_num}: {str(e)}")
        return None


def process_pdf(s3_path: str, temp_dir: str, output_dir: str, api_key: str) -> bool:
    """
    Process a single PDF from S3.

    Args:
        s3_path: S3 path to the PDF
        temp_dir: Directory for temporary files
        output_dir: Directory for output files
        api_key: OpenAI API key

    Returns:
        bool: True if the PDF has a table and was copied, False otherwise
    """
    # Extract filename from S3 path
    pdf_filename = os.path.basename(s3_path)
    local_pdf_path = os.path.join(temp_dir, pdf_filename)

    # Download PDF from S3
    if not download_pdf_from_s3(s3_path, local_pdf_path):
        return False

    pdf_filter = PdfFilter()

    if pdf_filter.filter_out_pdf(local_pdf_path):
        print(f"Filtering out {pdf_filename}")
        return False

    try:
        # Read the PDF to get the number of pages
        reader = pypdf.PdfReader(local_pdf_path)
        num_pages = len(reader.pages)

        if num_pages == 0:
            print(f"PDF {pdf_filename} has no pages")
            return False

        # Select a random page to check
        page_num = random.randint(0, num_pages - 1)
        page_num = random.choice([page_num, 0])  # Bias 50% of the time to do the first page

        # Check if the page contains a table
        result = check_for_table(local_pdf_path, page_num, api_key)

        if result is None:
            return False

        has_table, total_cells = result

        if has_table:
            # Get the cell count bucket for organizing output
            bucket_name = get_cell_count_bucket(total_cells)
            bucket_dir = os.path.join(output_dir, bucket_name)
            os.makedirs(bucket_dir, exist_ok=True)

            # Create output filename with basename_pgnum.pdf format
            pdf_basename = os.path.splitext(pdf_filename)[0]
            output_pdf_path = os.path.join(bucket_dir, f"{pdf_basename}_pg{page_num+1}.pdf")

            # Extract the single page
            writer = pypdf.PdfWriter()
            writer.add_page(reader.pages[page_num])

            # Write the output PDF
            with open(output_pdf_path, "wb") as output_file:
                writer.write(output_file)

            print(f"Extracted page {page_num+1} with table from {pdf_filename} to {bucket_name}/{os.path.basename(output_pdf_path)}")
            return True

        return False

    except Exception as e:
        print(f"Error processing {pdf_filename}: {str(e)}")
        return False
    finally:
        if os.path.exists(local_pdf_path):
            os.remove(local_pdf_path)


def main():
    parser = argparse.ArgumentParser(description="Identify and copy PDFs with tables")
    parser.add_argument("--input_list", required=True, help="Path to a file containing S3 paths to PDFs")
    parser.add_argument("--output_dir", required=True, help="Directory to copy PDFs with tables")
    parser.add_argument("--api_key", help="OpenAI API key (if not provided, will use OPENAI_API_KEY environment variable)")
    parser.add_argument("--temp_dir", default="/tmp/mine_tables", help="Directory for temporary files")
    parser.add_argument("--max_pdfs", type=int, default=100, help="Maximum number of PDFs with tables to find")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel workers (default: 1 for sequential)")
    parser.add_argument("--reservoir_multiplier", type=int, default=100, help="Multiplier for reservoir sampling (default: 100x max_pdfs)")
    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key not provided. Use --api_key or set OPENAI_API_KEY environment variable.")
        return

    os.makedirs(args.temp_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Reservoir sampling to get random subset of PDFs
    reservoir_size = args.max_pdfs * args.reservoir_multiplier
    pdf_paths = []
    n = 0  # Total number of items seen

    print(f"Using reservoir sampling with size {reservoir_size}")

    with open(args.input_list, "r") as f:
        for line in tqdm(f):
            n += 1
            path = line.strip()
            if not path:
                continue

            if len(pdf_paths) < reservoir_size:
                pdf_paths.append(path)
            else:
                # Randomly decide whether to include this item
                s = random.randint(1, n)
                if s <= reservoir_size:
                    pdf_paths[s - 1] = path

    # Shuffle the reservoir
    random.shuffle(pdf_paths)

    print(f"Sampled {len(pdf_paths)} PDF paths from {n} total paths")

    table_pdfs_found = 0

    if args.parallel > 1:
        # Parallel processing
        print(f"Processing PDFs with {args.parallel} parallel workers")

        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = []

            # Submit all tasks
            for s3_path in pdf_paths:
                if table_pdfs_found >= args.max_pdfs:
                    break
                future = executor.submit(process_pdf, s3_path, args.temp_dir, args.output_dir, api_key)
                futures.append(future)

            # Process results as they complete
            with tqdm(total=min(len(pdf_paths), args.max_pdfs), desc="Processing PDFs") as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            table_pdfs_found += 1
                            pbar.update(1)

                            if table_pdfs_found >= args.max_pdfs:
                                print(f"Reached maximum number of PDFs with tables ({args.max_pdfs}), stopping")
                                # Cancel remaining futures
                                for f in futures:
                                    f.cancel()
                                break
                    except Exception as e:
                        print(f"Error in parallel processing: {str(e)}")
    else:
        # Sequential processing
        for s3_path in tqdm(pdf_paths, desc="Processing PDFs"):
            if process_pdf(s3_path, args.temp_dir, args.output_dir, api_key):
                table_pdfs_found += 1

                if table_pdfs_found >= args.max_pdfs:
                    print(f"Reached maximum number of PDFs with tables ({args.max_pdfs}), stopping")
                    break

    print(f"Found and copied {table_pdfs_found} PDFs with tables to {args.output_dir}")


if __name__ == "__main__":
    main()
