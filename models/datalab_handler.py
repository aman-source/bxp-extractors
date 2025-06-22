import os
import mimetypes
import requests
import time
from pathlib import Path
import google.generativeai as genai

def detect_mime_type(file_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or "application/octet-stream"

def get_invoice_schema() -> str:
    return """
{
  "sellerInfo": {
    "name": "", "address1": "", "address2": "", "mobile": "", "phone": "", "gstin": "",
    "pan": "", "email": "", "city": "", "district": "", "state": "", "pincode": ""
  },
  "BuyerInfo": {
    "name": "", "address1": "", "address2": "", "mobile": "", "phone": "", "gstin": "",
    "pan": "", "aadhaar": "", "city": "", "district": "", "state": "", "pincode": ""
  },
  "shippedToInfo": {
    "name": "", "address1": "", "address2": "", "mobile": "", "phone": "", "gstin": "",
    "pan": "", "aadhaar": "", "city": "", "district": "", "state": "", "pincode": ""
  },
  "InvoiceDetails": {
    "invoiceDate": "", "invoiceNumber": "", "invoicetype": "", "invoiceMode": "", "transactionType": "",
    "invoiceDueDate": "", "quotationNumber": "", "quotationDate": "", "ewayNumber": "",
    "ewayDate": "", "ewayValidity": "", "IRN": "",
    "itemDetails": [
      {
        "itemName": "", "hsnCode": "", "quantity": "", "quantity1": "", "rate": "", "totalamount": "",
        "batchNumber": "", "discountPercent": "", "discountAmount": "", "freightAmount": "",
        "MfgDate": "", "ExpiryDate": "", "freeQty": "", "cgstRate": "", "sgstRate": "", "igstRate": "",
        "ItemtaxableAmount": "", "cgstAmount": "", "sgstAmount": "", "igstAmount": "", "Amount": ""
      }
    ],
    "charges": [ { "chargeName": "", "chargeAmount": "", "chargeType": "" } ],
    "invoiceSummary": {
      "taxableAmount": "", "totalCGST": "", "totalSGST": "", "totalIGST": "", "totalAmount": ""
    }
  },
  "accuracyLevel": "", "isDuplicate": ""
}

"""

def get_bank_statement_schema() -> str:
    return """
{
  "Countoftransactions": 0,
  "transactions": [
    {
      "date": "",
      "description": "",
      "ChequeNo": "",
      "debit": 0,
      "credit": 0
    }
  ]
}
"""

def get_schema_prompt(extracted_data, doc_type: str) -> str:
    schema = get_invoice_schema() if doc_type == "Invoice" else get_bank_statement_schema()
    return f"""
Given the following extracted data:

{extracted_data}

Restructure it into this JSON format. Leave any missing fields as empty strings:

{schema}

Only return valid JSON.
"""

def extract_from_datalab(file_path: Path, doc_type: str = "Invoice", poll_timeout: int = 600) -> dict:
    print(f"[DEBUG] Extracting from Datalab OCR: {file_path}")

    api_key = os.getenv("DATALAB_API_KEY")
    if not api_key:
        return {"error": "DATALAB_API_KEY not set in environment", "status": "failed"}

    mime_type = detect_mime_type(file_path)
    print(f"[DEBUG] Detected MIME type: {mime_type}")

    try:
        with open(file_path, "rb") as f:
            files = {
                'file': (file_path.name, f, mime_type),
                'langs': (None, "English")
            }
            headers = {"X-Api-Key": api_key}
            response = requests.post("https://www.datalab.to/api/v1/marker", files=files, headers=headers)

        if response.status_code != 200:
            print(f"[ERROR] Datalab API call failed: {response.status_code}")
            return {"error": response.text, "status": "failed"}

        initial_data = response.json()
        if not initial_data.get("request_check_url"):
            return {"error": "No check URL received", "status": "failed"}

        check_url = initial_data["request_check_url"]
        print(f"[DEBUG] Polling URL: {check_url}")

        poll_data = None

        for i in range(poll_timeout // 2):  # poll every 2s
            time.sleep(2)
            poll_response = requests.get(check_url, headers=headers)
            poll_data = poll_response.json()

            if poll_data.get("status") == "complete":
                print("[DEBUG] Datalab OCR completed.")

                # Step 2: Send to Gemini for restructuring
                try:
                    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
                    model = genai.GenerativeModel("gemini-1.5-flash")

                    gemini_response = model.generate_content([
                        str(poll_data),
                        get_schema_prompt(poll_data, doc_type)
                    ])

                    return {"result": gemini_response.text}

                except Exception as gemini_error:
                    return {"error": f"Gemini restructuring failed: {gemini_error}", "status": "failed"}

            print(f"[DEBUG] Poll attempt {i+1}: {poll_data.get('status')}")

        return {"error": "Polling timeout", "status": "failed"}

    except Exception as e:
        print(f"[ERROR] Exception during Datalab OCR: {e}")
        return {"error": str(e), "status": "failed"}