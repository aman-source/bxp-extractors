# utils/blob_utils.py
from azure.storage.blob import BlobServiceClient
import os

AZURE_CONN_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME")

def upload_to_blob(file, filename) -> str:
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONN_STR)
        blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER_NAME, blob=filename)

        file.seek(0)
        blob_client.upload_blob(file, overwrite=True)
        blob_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{AZURE_CONTAINER_NAME}/{filename}"
        return blob_url
    except Exception as e:
        print("Error uploading to blob:", e)
        return None
