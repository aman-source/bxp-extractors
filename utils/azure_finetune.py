import requests
import os
from dotenv import load_dotenv

load_dotenv()

def train_model_on_azure(training_container_sas_url, model_name=None, description=None):
    endpoint = os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT")
    api_key = os.getenv("AZURE_FORM_RECOGNIZER_KEY")

    headers = {
        "Ocp-Apim-Subscription-Key": api_key,
        "Content-Type": "application/json"
    }

    payload = {
        "modelId": model_name or "finetune-model-" + os.urandom(4).hex(),
        "description": description or "Auto-trained model from fine-tune UI",
        "buildMode": "template",  # or "neural" if you're using the new model types
        "azureBlobSource": {
            "containerUrl": training_container_sas_url
        }
    }

    url = f"{endpoint}/formrecognizer/documentModels:build?api-version=2023-07-31"

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 202:
        result = response.json()
        model_id = result.get("modelId")
        print(f"Model training started. Model ID: {model_id}")
        return model_id
    else:
        print("❌ Failed to trigger training:", response.status_code, response.text)
        return None
