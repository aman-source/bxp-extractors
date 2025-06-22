import requests
import json
import base64

from utils.blob_utils import upload_to_blob

API_BASE_URL = "https://qa.pilloo.ai/api/Pilloo"
API_KEY = "JmSwTvNvDgKrPrRmSpSbJb"
HEADERS = {
    "x-api-key": API_KEY,
    "Content-Type": "application/json"
}

def file_to_base64(file):
    file.seek(0)
    return base64.b64encode(file.read()).decode('utf-8')

def save_file_record(file, filename, model, doc_type, expected_json="", status="Pending", file_type="Pdf"):
    try:
        blob_url = upload_to_blob(file, filename)
        if not blob_url:
            print("[ERROR] Blob upload failed.")
            return None

        file_b64 = file_to_base64(file)
        payload = {
            "fileType": file_type,
            "docType": doc_type,
            "status": status,
            "model": model,
            "expectedJson": json.dumps(expected_json),
            "fileBase64": file_b64,
            "fileUrl": blob_url
        }

        response = requests.post(f"{API_BASE_URL}/UploadDocForFineTuning", headers=HEADERS, json=payload)
        data = response.json()

        if response.status_code == 200 and data.get("message") == "Success":
            return blob_url
        else:
            print("Upload failed:", data)
            return None
    except Exception as e:
        print("[EXCEPTION] Upload error:", e)
        return None

def get_finetune_files():
    """Fetch list of docs marked as 'needs_finetune' from external API."""
    try:
        response = requests.get(f"{API_BASE_URL}/GetDocsForFineTuning", headers=HEADERS)
        data = response.json()
        if response.status_code == 200 and data.get("message") == "Success" and 'data' in data:
            return data['data'] 
        else:
            print("Fetch error:", data.get("message"))
            return []
    except Exception as e:
        print("Exception fetching files:", e)
        return []
