
import uuid
from pathlib import Path

def detect_mime_type(file_path_or_name: str) -> str:
    ext = str(file_path_or_name).lower().split(".")[-1]
    return {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "pdf": "application/pdf"
    }.get(ext, None)


def save_uploaded_file(uploaded_file):
    out_dir = Path("temp_uploads")
    out_dir.mkdir(exist_ok=True)
    ext = uploaded_file.name.split('.')[-1]
    file_path = out_dir / f"{uuid.uuid4()}.{ext}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    return file_path
