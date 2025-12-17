"""
FastAPI service for IT Risk Analysis
Loads the trained model and provides risk classification endpoints.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
import os
from typing import Literal

# Initialize FastAPI app
app = FastAPI(
    title="IT Risk Analysis API",
    description="Classify risk level of cloud providers based on various metrics",
    version="1.0.0"
)

# Model paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'it_risk_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'it_risk_scaler.pkl')
PROVIDER_ENCODER_PATH = os.path.join(MODEL_DIR, 'it_risk_provider_encoder.pkl')
RISK_ENCODER_PATH = os.path.join(MODEL_DIR, 'it_risk_encoder.pkl')

# Global variables for model and preprocessing objects
model = None
scaler = None
provider_encoder = None
risk_encoder = None

def load_model():
    """Load the trained model and preprocessing objects"""
    global model, scaler, provider_encoder, risk_encoder
    
    required_files = {
        "Model": MODEL_PATH,
        "Scaler": SCALER_PATH,
        "Provider Encoder": PROVIDER_ENCODER_PATH,
        "Risk Encoder": RISK_ENCODER_PATH
    }
    
    for name, path in required_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{name} file not found at {path}. "
                "Please run it_risk_model.py first to train and save the model."
            )
    
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    provider_encoder = joblib.load(PROVIDER_ENCODER_PATH)
    risk_encoder = joblib.load(RISK_ENCODER_PATH)
    print("Model and preprocessing objects loaded successfully!")

# Load model on startup
@app.on_event("startup")
async def startup_event():
    """Load model when the API starts"""
    try:
        load_model()
    except Exception as e:
        print(f"Warning: Could not load model on startup: {e}")
        print("Model will be loaded on first prediction request.")

# Request/Response models
class CloudProviderMetrics(BaseModel):
    """Input metrics for cloud provider risk assessment"""
    provider: Literal['AWS', 'Azure', 'GCP', 'IBM Cloud', 'Oracle Cloud', 'Other'] = Field(
        ..., description="Cloud provider name"
    )
    uptime_percentage: float = Field(
        ..., ge=0, le=100, description="Uptime percentage (0-100)"
    )
    security_incidents: int = Field(
        ..., ge=0, description="Number of security incidents in the past year"
    )
    compliance_score: float = Field(
        ..., ge=0, le=100, description="Compliance score (0-100)"
    )
    cost_efficiency: float = Field(
        ..., ge=0, le=100, description="Cost efficiency score (0-100)"
    )
    support_rating: float = Field(
        ..., ge=0, le=5, description="Support rating (0-5)"
    )
    data_encryption: int = Field(
        ..., ge=0, le=1, description="Data encryption enabled (0=No, 1=Yes)"
    )
    backup_frequency_days: int = Field(
        ..., ge=1, description="Backup frequency in days"
    )
    response_time_hours: float = Field(
        ..., ge=0, description="Average incident response time in hours"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "provider": "AWS",
                "uptime_percentage": 99.9,
                "security_incidents": 2,
                "compliance_score": 95.0,
                "cost_efficiency": 88.5,
                "support_rating": 4.5,
                "data_encryption": 1,
                "backup_frequency_days": 1,
                "response_time_hours": 2.0
            }
        }

class RiskPrediction(BaseModel):
    """Response model for risk prediction"""
    provider: str
    risk_level: str = Field(..., description="Predicted risk level: Low, Medium, or High")
    risk_probability: dict = Field(..., description="Probability for each risk level")
    
    class Config:
        schema_extra = {
            "example": {
                "provider": "AWS",
                "risk_level": "Low",
                "risk_probability": {
                    "Low": 0.85,
                    "Medium": 0.12,
                    "High": 0.03
                }
            }
        }

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool

# Endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "IT Risk Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "/analyze",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check if the API and model are loaded"""
    return {
        "status": "healthy" if model is not None else "model not loaded",
        "model_loaded": model is not None
    }

@app.post("/analyze", response_model=RiskPrediction, tags=["Risk Analysis"])
async def analyze_risk(metrics: CloudProviderMetrics):
    """
    Analyze cloud provider risk based on input metrics
    
    Returns the predicted risk level (Low, Medium, High) and probability distribution
    """
    global model, scaler, provider_encoder, risk_encoder
    
    # Load model if not loaded
    if any(obj is None for obj in [model, scaler, provider_encoder, risk_encoder]):
        try:
            load_model()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Model not available: {str(e)}"
            )
    
    try:
        # Encode provider
        try:
            provider_encoded = provider_encoder.transform([metrics.provider])[0]
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown provider: {metrics.provider}. Valid providers: {list(provider_encoder.classes_)}"
            )
        
        # Prepare input data
        input_data = pd.DataFrame({
            'provider_encoded': [provider_encoded],
            'uptime_percentage': [metrics.uptime_percentage],
            'security_incidents': [metrics.security_incidents],
            'compliance_score': [metrics.compliance_score],
            'cost_efficiency': [metrics.cost_efficiency],
            'support_rating': [metrics.support_rating],
            'data_encryption': [metrics.data_encryption],
            'backup_frequency_days': [metrics.backup_frequency_days],
            'response_time_hours': [metrics.response_time_hours]
        })
        
        # Scale and predict
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Decode risk category
        risk_level = risk_encoder.inverse_transform([prediction])[0]
        
        # Create probability dictionary
        risk_probability = {
            risk_class: float(prob)
            for risk_class, prob in zip(risk_encoder.classes_, prediction_proba)
        }
        
        return RiskPrediction(
            provider=metrics.provider,
            risk_level=risk_level,
            risk_probability=risk_probability
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Risk analysis failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
