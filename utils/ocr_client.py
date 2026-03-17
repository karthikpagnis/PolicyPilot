"""OCR client using pytesseract for text extraction from images."""

import pytesseract
from PIL import Image
import io
import base64


def extract_text_from_base64_image(b64_image: str) -> str:
    """
    Extract text from a base64-encoded image using pytesseract (Tesseract OCR).

    Args:
        b64_image: Base64-encoded image string

    Returns:
        Extracted text from the image
    """
    try:
        # Decode base64 to bytes
        image_bytes = base64.b64decode(b64_image)

        # Open image with PIL
        image = Image.open(io.BytesIO(image_bytes))

        # Extract text using pytesseract
        text = pytesseract.image_to_string(image)

        return text.strip()
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""


def extract_text_from_image_file(image_path: str) -> str:
    """
    Extract text from an image file using pytesseract.

    Args:
        image_path: Path to image file

    Returns:
        Extracted text from the image
    """
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""
