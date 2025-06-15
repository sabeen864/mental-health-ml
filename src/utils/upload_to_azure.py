from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
import os

print("Starting script...")  # Debug
# Load environment variables
load_dotenv()
CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
print(f"Connection string loaded: {CONNECTION_STRING[:20]}...")  # Debug
CONTAINER_NAME = "models"
MODEL_PATH = "../models/LogisticRegression_model.pkl"
DATASET_PATH = "data\Cleaned\cleaned_mental_health.csv"
BLOB_MODEL_NAME = "LogisticRegression_model.pkl"
BLOB_DATASET_NAME = "cleaned_mental_health.csv"

def upload_to_blob_storage():
    try:
        print("Initializing BlobServiceClient...")  # Debug
        # Initialize BlobServiceClient
        blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        print(f"Connected to container: {CONTAINER_NAME}")  # Debug

        # Upload model
        print(f"Uploading {MODEL_PATH}...")  # Debug
        with open(MODEL_PATH, "rb") as model_file:
            container_client.upload_blob(name=BLOB_MODEL_NAME, data=model_file, overwrite=True)
        print(f"Uploaded {BLOB_MODEL_NAME} to Azure Blob Storage")

        # Upload dataset
        print(f"Uploading {DATASET_PATH}...")  # Debug
        with open(DATASET_PATH, "rb") as dataset_file:
            container_client.upload_blob(name=BLOB_DATASET_NAME, data=dataset_file, overwrite=True)  # Fixed
        print(f"Uploaded {BLOB_DATASET_NAME} to Azure Blob Storage")

    except Exception as e:
        print(f"Error uploading to Azure Blob Storage: {str(e)}")  # Debug

if __name__ == "__main__":
    print("Running upload_to_blob_storage...")  # Debug
    upload_to_blob_storage()