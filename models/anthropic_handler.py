import os
import base64
import mimetypes
from pathlib import Path
import anthropic

def encode_file_base64(file_path: Path) -> tuple[str, str]:
    """Returns (media_type, base64_encoded_string) for a given file."""
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        mime_type = "application/octet-stream"

    with open(file_path, "rb") as f:
        encoded_data = base64.b64encode(f.read()).decode("utf-8")

    return mime_type, encoded_data

def get_invoice_schema_prompt() -> str:
    return """
You are an expert in invoice extraction. Given the attached document, extract and return a JSON object
with the following structured format:

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

Only return valid JSON. Leave missing fields as empty strings.
"""

def get_bank_statement_schema_prompt() -> str:
    return """
You are an expert in bank statement extraction. Given the attached document (which may contain multiple pages),
return a JSON object in the following format:

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

Rules:
- Return only valid JSON
- Countoftransactions must be the length of the `transactions` array
- Leave any missing values as empty strings
- No extra text — only the JSON output
"""

def extract_from_claude(file_path: Path, doc_type: str = "Invoice") -> dict:
    """Extracts structured data from a local PDF or image using Claude 3 via base64 input."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return {"error": "ANTHROPIC_API_KEY is not set", "status": "failed"}

    try:
        media_type, base64_data = encode_file_base64(file_path)
    except Exception as e:
        return {"error": f"File read/encode failed: {e}", "status": "failed"}

    try:
        prompt = get_invoice_schema_prompt() if doc_type == "Invoice" else get_bank_statement_schema_prompt()

        client = anthropic.Anthropic(api_key=api_key)

        message = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_data
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )
        return {"result": message.content}
    except Exception as e:
        return {"error": str(e), "status": "failed"}
