import os
from pathlib import Path
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
import google.generativeai as genai

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

def extract_from_azure_finetuned(file_path: Path, doc_type: str = "Invoice", model_id: str = "bookxpert-extractor-v0.1") -> dict:
    print(f"[DEBUG] Extracting with Azure fine-tuned model {model_id} for {doc_type}: {file_path}")

    endpoint = os.getenv("AZURE_FORMRECOGNIZER_ENDPOINT")
    key = os.getenv("AZURE_FORMRECOGNIZER_KEY")

    if not endpoint or not key:
        return {"error": "Azure credentials not set in environment", "status": "failed"}

    try:
        client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    except Exception as e:
        return {"error": f"Client initialization failed: {e}", "status": "failed"}

    try:
        with open(file_path, "rb") as f:
            poller = client.begin_analyze_document(model_id=model_id, body=f)
            result = poller.result()
    except Exception as e:
        return {"error": f"Azure analysis failed: {e}", "status": "failed"}

    try:
        documents = []
        for doc in result.documents:
            doc_data = {}
            for name, field in doc.fields.items():
                try:
                    # Handle value based on type
                    field_value = getattr(field, "value", None)
                    if field_value is None:
                        field_value = getattr(field, "content", "")

                    doc_data[name] = {
                        "value": field_value,
                        "confidence": field.confidence if hasattr(field, "confidence") else None
                    }
                except Exception as inner_e:
                    doc_data[name] = {
                        "value": f"[Error reading field: {inner_e}]",
                        "confidence": None
                    }

            documents.append(doc_data)

        print("[DEBUG] Azure fine-tuned document parsed")

        try:
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            model = genai.GenerativeModel("gemini-1.5-flash")

            gemini_response = model.generate_content([
                str(documents[0]),
                get_schema_prompt(documents[0], doc_type)
            ])

            return {"result": gemini_response.text}

        except Exception as gemini_error:
            return {"error": f"Gemini restructuring failed: {gemini_error}", "status": "failed"}

    except Exception as e:
        return {"error": f"Failed to parse Azure result: {e}", "status": "failed"}