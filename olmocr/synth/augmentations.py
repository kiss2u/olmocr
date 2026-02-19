import os

from PIL import Image

from olmocr.synth.claude_client import (
    DEFAULT_MODEL_NAME,
    claude_stream,
    extract_code_block,
)


async def densify_html(client, html_content):
    """Call Claude API to generate a denser version of HTML content by doubling information density."""
    import olmocr.synth.mine_html_templates as _mine

    try:
        dense_response = await claude_stream(
            client,
            model=DEFAULT_MODEL_NAME,
            max_tokens=50000,
            temperature=0.7,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": html_content
                        },


                        {
                            "type": "text",
                            "text": "The HTML above describes a webpage meant to render into a single printed PDF page. Please output a new full synthetic webpage that increases the amount of information on this page by 2X. "
                            "Your goal is to shrink the font size and add more synthetic content so that the general idea and structure of the page is preserved, but so that it contains twice as many final tokens. "
                            "Be careful to adjust any elements (such as footers) so that they will not overlap the main body of the newly expanded document. "
                            "But remember that it still needs to render as a single static HTML page that will print out to ONE page on a printer or in PDF form. "
                            "Output the complete revised HTML in a ```html code block."
                        },
                    ],
                }
            ],
        )

        dense_html_text = ""
        for content in dense_response.content:
            if content.type == "text":
                dense_html_text += content.text

        # Track token usage on the main module's globals
        if hasattr(dense_response, "usage"):
            _mine.total_input_tokens += dense_response.usage.input_tokens
            _mine.total_output_tokens += dense_response.usage.output_tokens

        dense_html = extract_code_block(dense_html_text)
        if not dense_html:
            print("Warning: No HTML code block found in densifying response")
            return None

        return dense_html

    except Exception as e:
        print(f"Error calling Claude API: {e}")
        return None


def apply_jpeg_compression(pdf_path, quality, temp_dir):
    """
    Apply JPEG compression to a PDF by converting to PNG, then to JPEG, then back to PDF.

    Args:
        pdf_path: Path to the input PDF file
        quality: JPEG quality level (70-95)
        temp_dir: Directory for temporary files

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import base64
        import io

        from olmocr.data.renderpdf import render_pdf_to_base64png

        # Create temp file paths
        temp_jpeg_path = os.path.join(temp_dir, "temp_page.jpg")
        temp_pdf_path = os.path.join(temp_dir, "temp_compressed.pdf")

        # Render at high resolution for better quality
        png_base64 = render_pdf_to_base64png(pdf_path, 1, 1288)

        # Decode base64 PNG data
        png_data = base64.b64decode(png_base64)
        png_buffer = io.BytesIO(png_data)

        # Open the PNG and convert to JPEG with specified quality
        with Image.open(png_buffer) as img:
            # Convert RGBA to RGB if necessary
            if img.mode in ('RGBA', 'LA', 'P'):
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                # Paste using alpha channel as mask if available
                if img.mode == 'RGBA' or img.mode == 'LA':
                    rgb_img.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else img.split()[1])
                else:
                    rgb_img.paste(img)
                img = rgb_img

            # Save as JPEG with specified quality
            img.save(temp_jpeg_path, 'JPEG', quality=quality, optimize=True)

            # Convert JPEG back to PDF
            img_for_pdf = Image.open(temp_jpeg_path)
            img_for_pdf.save(temp_pdf_path, 'PDF', resolution=100.0)

        # Replace original PDF with compressed version
        os.replace(temp_pdf_path, pdf_path)

        # Clean up temp files
        if os.path.exists(temp_jpeg_path):
            os.remove(temp_jpeg_path)

        return True

    except Exception as e:
        print(f"Error applying JPEG compression: {e}")
        return False
