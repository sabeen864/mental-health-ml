from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import warnings
import logging
from src.utils.download_from_azure import download_from_blob_storage

from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter

# --- Configure logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- Configure OpenTelemetry for Azure Monitor ---
resource = Resource(attributes={"service.name": "mental-health-ml-api"})
trace.set_tracer_provider(TracerProvider(resource=resource))

exporter = AzureMonitorTraceExporter(
    connection_string=os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
)

span_processor = BatchSpanProcessor(exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

app = FastAPI()
FastAPIInstrumentor.instrument_app(app)

# --- Suppress warnings ---
warnings.filterwarnings('ignore')

# --- Download model if not present ---
try:
    download_from_blob_storage()
    logger.info("Model download attempted (if required).")
except Exception as e:
    logger.error(f"Error during model download: {e}")
    raise

# --- Load model ---
try:
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "LogisticRegression_model.pkl"))
    model = joblib.load(model_path)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

@app.get("/")
async def read_root():
    logger.info("Root endpoint accessed.")
    return {"message": "Welcome to Mental Health Prediction API! Use /predict for predictions and /health for status."}

@app.get("/health")
async def health_check():
    logger.info("Health check requested.")
    return {"status": "healthy"}

# --- Input Schema ---
class InputData(BaseModel):
    Age: int
    Gender: str
    family_history: str
    benefits: str
    care_options: str
    anonymity: str
    leave: str
    work_interfere: str
    self_employed: str
    remote_work: str

# --- Encoders setup ---
le_dict = {}
categories = [
    'Male', 'Female', 'Other', 'Yes', 'No',
    "Don't know", 'Never', 'Rarely', 'Sometimes', 'Often'
]

cat_columns = [
    'Gender', 'family_history', 'benefits', 'care_options',
    'anonymity', 'leave', 'work_interfere', 'self_employed', 'remote_work'
]

for col in cat_columns:
    le = LabelEncoder()
    le.fit(categories)
    le_dict[col] = le

@app.post("/predict")
async def predict(data: InputData):
    logger.info(f"Prediction requested with data: {data.dict()}")
    try:
        input_df = pd.DataFrame([data.dict()])

        # Encode categorical columns
        for col in cat_columns:
            input_df[col] = le_dict[col].transform(input_df[col])

        # Align columns with model
        input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

        prediction = model.predict(input_df)[0]
        logger.info(f"Prediction completed: {prediction}")

        return {"prediction": "Yes" if prediction == 1 else "No"}

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed. Check input data or server logs.")
