import joblib
import json
import logging
import time
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from typing import Literal

# --- Boilerplate for Logging and Tracing ---
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

# Tracing setup
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = BatchSpanProcessor(CloudTraceSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

# Logging setup
logger = logging.getLogger("ml-service")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger.addHandler(handler)
# --- End Boilerplate ---


# Load model and preprocessing artifacts
model = joblib.load("heart_disease_model.joblib")
imputer = joblib.load("imputer.joblib")
label_encoder = joblib.load("label_encoder.joblib")
training_columns = joblib.load("training_columns.joblib")
app = FastAPI()

# Define the input data model using Pydantic
class HeartDiseaseInput(BaseModel):
    age: int
    gender: Literal['male', 'female']
    cp: int = Field(..., description="Chest Pain type")
    trestbps: float = Field(..., description="Resting blood pressure")
    chol: float = Field(..., description="Serum cholesterol in mg/dl")
    fbs: int = Field(..., description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., description="Resting electrocardiographic results")
    thalach: float = Field(..., description="Maximum heart rate achieved")
    exang: int = Field(..., description="Exercise induced angina")
    oldpeak: float = Field(..., description="ST depression induced by exercise relative to rest")
    slope: int = Field(..., description="Slope of the peak exercise ST segment")
    ca: int = Field(..., description="Number of major vessels colored by fluoroscopy")
    thal: int = Field(..., description="Thalium stress test result")

@app.get("/health")
def health():
    """Health check endpoint for Kubernetes probes."""
    return {"status": "ok"}

@app.post("/predict")
async def predict(input: HeartDiseaseInput, request: Request):
    """Prediction endpoint with logging and tracing."""
    with tracer.start_as_current_span("inference") as span:
        start = time.time()

        input_df = pd.DataFrame([input.dict()])
        
        input_df['gender'] = input_df['gender'].map({'male': 1, 'female': 0})
        
        # Reorder the DataFrame columns to match the training order
        input_df_reordered = input_df.reindex(columns=training_columns)
        
        # Use the reordered DataFrame for imputation and prediction
        input_processed = pd.DataFrame(imputer.transform(input_df_reordered), columns=training_columns)
        
        pred_encoded = model.predict(input_processed)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]

        latency = round((time.time() - start) * 1000, 2)
        trace_id = format(span.get_span_context().trace_id, "032x")

        logger.info(json.dumps({
            "trace_id": trace_id,
            "input": input.dict(),
            "prediction": pred_label,
            "latency_ms": latency
        }))

        return {"prediction": pred_label, "latency_ms": latency}
