from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
import os

print("Starting download script...")
load_dotenv()
CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
print(f"Connection String: {CONNECTION_STRING}")  # Debug
if not CONNECTION_STRING:
    raise ValueError("AZURE_CONNECTION_STRING is not set")
CONTAINER_NAME = "models"
BLOB_MODEL_NAME = "LogisticRegression_model.pkl"
MODEL_DEST_PATH = "models/LogisticRegression_model.pkl"

def download_from_blob_storage():
    try:
        print("Initializing BlobServiceClient...")
        blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        print(f"Connected to container: {CONTAINER_NAME}")

        os.makedirs("models", exist_ok=True)
        print(f"Downloading {BLOB_MODEL_NAME}...")
        with open(MODEL_DEST_PATH, "wb") as model_file:
            blob_client = container_client.get_blob_client(BLOB_MODEL_NAME)
            model_file.write(blob_client.download_blob().readall())
        print(f"Downloaded {BLOB_MODEL_NAME} to {MODEL_DEST_PATH}")

    except Exception as e:
        print(f"Error downloading from Azure Blob Storage: {str(e)}")