import os
from pathlib import Path
import google.generativeai as genai

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

Only return valid JSON. Strictly leave missing fields as empty strings.
"""

def get_bank_statement_schema_prompt() -> str:
    return """
You are an expert in bank statement extraction. Given the attached document (which may contain multiple pages),
return a valid JSON object in the following format:

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

def extract_from_gemini(file_path: Path, doc_type: str = "Invoice") -> dict:
    try:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

        uploaded_file = genai.upload_file(path=str(file_path))
        print(f"[DEBUG] Uploaded file name: {uploaded_file.display_name}")

        model = genai.GenerativeModel("models/gemini-1.5-flash")

        prompt = (
            get_invoice_schema_prompt()
            if doc_type == "Invoice"
            else get_bank_statement_schema_prompt()
        )

        response = model.generate_content([
            uploaded_file,
            prompt
        ])

        return {"result": response.text}
    except Exception as e:
        return {"error": str(e), "status": "failed"}
