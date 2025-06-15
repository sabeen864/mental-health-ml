from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import warnings
from src.utils.download_from_azure import download_from_blob_storage

warnings.filterwarnings('ignore')

# Download model from Azure
download_from_blob_storage()
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "LogisticRegression_model.pkl"))
model = joblib.load(model_path)

# Initialize FastAPI
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to Mental Health Prediction API! Use /predict for predictions and /health for status."}

@app.get("/health")
def health_check():
    return {"status": "Model is up and running!"}

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

le_dict = {}
for col in ['Gender', 'family_history', 'benefits', 'care_options', 'anonymity', 'leave', 'work_interfere', 'self_employed', 'remote_work']:
    le_dict[col] = LabelEncoder()
    le_dict[col].fit(['Male', 'Female', 'Other', 'Yes', 'No', "Don't know", 'Never', 'Rarely', 'Sometimes', 'Often'])

@app.post("/predict")
def predict(data: InputData):
    input_df = pd.DataFrame([data.dict()])
    for col in ['Gender', 'family_history', 'benefits', 'care_options',
                'anonymity', 'leave', 'work_interfere', 'self_employed', 'remote_work']:
        input_df[col] = le_dict[col].transform(input_df[col])
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
    prediction = model.predict(input_df)[0]
    return {"prediction": "Yes" if prediction == 1 else "No"}