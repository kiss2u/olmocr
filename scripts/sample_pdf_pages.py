#!/usr/bin/env python3
"""
Sample random pages from PDFs in an input directory and save them as individual PDFs.

This script takes PDFs from an input directory, randomly selects N pages from each,
and writes each selected page as a separate PDF file in the output directory.
Uses a fixed random seed for reproducibility.
"""

import argparse
import random
from pathlib import Path

from pypdf import PdfReader, PdfWriter


def sample_pdf_pages(input_dir: Path, output_dir: Path, num_pages: int, seed: int = 42):
    """
    Sample random pages from PDFs and save them as individual files.

    Args:
        input_dir: Directory containing input PDF files
        output_dir: Directory to save sampled pages
        num_pages: Number of pages to sample from each PDF
        seed: Random seed for reproducibility
    """
    # Set random seed for consistency
    random.seed(seed)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all PDF files from input directory
    pdf_files = sorted(input_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return

    print(f"Found {len(pdf_files)} PDF file(s)")

    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path.name}")

        try:
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)

            print(f"  Total pages: {total_pages}")

            # Determine how many pages to sample (min of requested and available)
            pages_to_sample = min(num_pages, total_pages)

            if pages_to_sample < num_pages:
                print(f"  Warning: Only {total_pages} pages available, sampling all")

            # Randomly select page indices
            selected_indices = sorted(random.sample(range(total_pages), pages_to_sample))
            print(f"  Selected page indices: {selected_indices}")

            # Extract and save each selected page
            for page_idx in selected_indices:
                # Create writer for single page
                writer = PdfWriter()
                writer.add_page(reader.pages[page_idx])

                # Create output filename: original_name_page_N.pdf
                output_filename = f"{pdf_path.stem}_page_{page_idx}.pdf"
                output_path = output_dir / output_filename

                # Write the single page PDF
                with open(output_path, "wb") as output_file:
                    writer.write(output_file)

                print(f"    Saved: {output_filename}")

        except Exception as e:
            print(f"  Error processing {pdf_path.name}: {e}")
            continue

    print(f"\nDone! Output saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Sample random pages from PDFs and save as individual files")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing input PDF files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save sampled pages")
    parser.add_argument("--pages", type=int, required=True, help="Number of pages to sample from each PDF")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")

    args = parser.parse_args()

    # Validate arguments
    if not args.input_dir.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return

    if not args.input_dir.is_dir():
        print(f"Error: Input path is not a directory: {args.input_dir}")
        return

    if args.pages < 1:
        print(f"Error: Number of pages must be at least 1")
        return

    sample_pdf_pages(args.input_dir, args.output_dir, args.pages, args.seed)


if __name__ == "__main__":
    main()
