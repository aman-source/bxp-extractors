# models/mistral_handler.py

import base64
import os
from pathlib import Path
from mistralai import Mistral

def detect_mime_type(file_path: Path) -> str:
    """
    Detect the MIME type based on the file extension.
    """
    ext = file_path.suffix.lower().lstrip(".")
    return {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "pdf": "application/pdf"
    }.get(ext, None)

def extract_from_mistral(file_path: Path) -> dict:
    """
    Extract structured data from a PDF or image using Mistral OCR API.

    Args:
        file_path (Path): Path to the saved file.

    Returns:
        dict: OCR result or error message.
    """
    # Read API key from environment
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        return {"error": "MISTRAL_API_KEY is not set in environment", "status": "failed"}

    # Detect file type
    mime_type = detect_mime_type(file_path)
    if not mime_type:
        return {"error": f"Unsupported file extension for: {file_path.name}", "status": "failed"}

    # Read file and encode to base64
    try:
        with open(file_path, "rb") as f:
            base64_data = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        return {"error": f"Failed to read file: {e}", "status": "failed"}


    if mime_type == "application/pdf":
        document = {
            "type": "document_url",
            "document_url": f"data:{mime_type};base64,{base64_data}"
        }
    else:
        document = {
            "type": "image_url",
            "image_url": f"data:{mime_type};base64,{base64_data}"
        }

    # Call Mistral OCR API
    try:
        client = Mistral(api_key=api_key)
        response = client.ocr.process(
            model="mistral-ocr-latest",
            document=document
        )
        return response
    except Exception as e:
        return {"error": str(e), "status": "failed"}
