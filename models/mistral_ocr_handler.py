import os
from pathlib import Path
from mistralai import Mistral

def detect_mime_type(file_path: Path) -> str:
    ext = file_path.suffix.lower().lstrip(".")
    mime_type = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "pdf": "application/pdf"
    }.get(ext, None)
    print(f"[DEBUG] Detected MIME type: {mime_type} for extension: {ext}")
    return mime_type

def get_invoice_schema_prompt() -> str:
    return (
        "You are an expert in invoice extraction. Given the attached document, extract and return a JSON object "
        "with the following structured format:\n\n"
        "{\n"
        "  sellerInfo: {name, address1, address2, mobile, phone, gstin, pan, email, city, district, state, pincode},\n"
        "  BuyerInfo: {name, address1, address2, mobile, phone, gstin, pan, aadhaar, city, district, state, pincode},\n"
        "  shippedToInfo: {name, address1, address2, mobile, phone, gstin, pan, aadhaar, city, district, state, pincode},\n"
        "  InvoiceDetails: {\n"
        "    invoiceDate, invoiceNumber, invoicetype, invoiceMode, transactionType, invoiceDueDate,\n"
        "    quotationNumber, quotationDate, ewayNumber, ewayDate, ewayValidity, IRN,\n"
        "    itemDetails: [ {itemName, hsnCode, quantity, quantity1, rate, totalamount, batchNumber, discountPercent,\n"
        "                  discountAmount, freightAmount, MfgDate, ExpiryDate, freeQty, cgstRate, sgstRate, igstRate,\n"
        "                  ItemtaxableAmount, cgstAmount, sgstAmount, igstAmount, Amount} ],\n"
        "    charges: [ {chargeName, chargeAmount, chargeType} ],\n"
        "    invoiceSummary: {taxableAmount, totalCGST, totalSGST, totalIGST, totalAmount}\n"
        "  },\n"
        "  accuracyLevel, isDuplicate\n"
        "}\n\n"
        "If a field is missing, return an empty string. Only return the JSON output."
    )

def get_bank_statement_schema_prompt() -> str:
    return (
        "You are an expert in bank statement extraction. Given the attached document (which may be multi-page), "
        "extract and return a valid JSON object in the following format:\n\n"
        "{\n"
        "  Countoftransactions: 0,\n"
        "  transactions: [\n"
        "    {date: '', description: '', ChequeNo: '', debit: 0, credit: 0}\n"
        "  ]\n"
        "}\n\n"
        "- Return all transactions from the document\n"
        "- Countoftransactions must match the length of the `transactions` array\n"
        "- Leave missing values as empty strings\n"
        "- Strictly return only valid JSON — no extra text"
    )

def extract_from_mistral(file_path: Path, doc_type: str = "Invoice") -> dict:
    print(f"[DEBUG] Extracting from file (LLM Q&A): {file_path}")

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        return {"error": "MISTRAL_API_KEY not set.", "status": "failed"}

    client = Mistral(api_key=api_key)

    try:
        print("[DEBUG] Uploading file to Mistral...")
        uploaded_file = client.files.upload(file={
            "file_name": file_path.name,
            "content": open(file_path, "rb"),
        }, purpose="ocr")
        signed_url = client.files.get_signed_url(file_id=uploaded_file.id)
        print(f"[DEBUG] File uploaded. Signed URL: {signed_url.url}")
    except Exception as e:
        return {"error": f"Failed to upload file: {e}", "status": "failed"}

    # Choose prompt based on doc_type
    schema_prompt = (
        get_invoice_schema_prompt()
        if doc_type == "Invoice"
        else get_bank_statement_schema_prompt()
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": schema_prompt
                },
                {
                    "type": "document_url",
                    "document_url": signed_url.url
                }
            ]
        }
    ]

    try:
        print("[DEBUG] Sending chat completion request to Mistral...")
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=messages
        )
        print("[DEBUG] Response received.")
        return {"result": response.choices[0].message.content}
    except Exception as e:
        return {"error": str(e), "status": "failed"}
