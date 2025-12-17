"""
Home Price Prediction API
FastAPI application that serves home price predictions using a trained ML model.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import numpy as np
from typing import Optional

# Initialize FastAPI app
app = FastAPI(
    title="Home Price Predictor API",
    description="API to predict home prices based on property features",
    version="1.0.0"
)

# Models directory
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

# Load trained model, scaler, and encoder
try:
    model = joblib.load(os.path.join(MODELS_DIR, 'home_price_model.pkl'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'home_price_scaler.pkl'))
    condition_encoder = joblib.load(os.path.join(MODELS_DIR, 'condition_encoder.pkl'))
except FileNotFoundError as e:
    raise RuntimeError(f"Model files not found. Ensure models are trained first: {e}")


# Request/Response Models
class HomePriceRequest(BaseModel):
    """Request model for home price prediction"""
    bedrooms: int
    bathrooms: int
    sqft: int
    zip_code: int
    condition: str

    class Config:
        json_schema_extra = {
            "example": {
                "bedrooms": 3,
                "bathrooms": 2,
                "sqft": 1500,
                "zip_code": 90210,
                "condition": "Good"
            }
        }


class HomePriceResponse(BaseModel):
    """Response model for home price prediction"""
    predicted_price: float
    bedrooms: int
    bathrooms: int
    sqft: int
    zip_code: int
    condition: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=True
    )


@app.post("/predict", response_model=HomePriceResponse)
async def predict_price(request: HomePriceRequest):
    """
    Predict home price based on property features
    
    - **bedrooms**: Number of bedrooms (1-5)
    - **bathrooms**: Number of bathrooms (1-3)
    - **sqft**: Square footage (800-4000)
    - **zip_code**: Zip code
    - **condition**: Property condition (Excellent, Good, Fair, Remodel)
    """
    try:
        # Validate condition
        try:
            condition_encoded = condition_encoder.transform([request.condition])[0]
        except ValueError:
            valid_conditions = list(condition_encoder.classes_)
            raise HTTPException(
                status_code=400,
                detail=f"Invalid condition. Valid options: {valid_conditions}"
            )
        
        # Prepare features for prediction
        features = np.array([[
            request.bedrooms,
            request.bathrooms,
            request.sqft,
            request.zip_code,
            condition_encoded
        ]], dtype=float)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        predicted_price = model.predict(features_scaled)[0]
        
        return HomePriceResponse(
            predicted_price=float(predicted_price),
            bedrooms=request.bedrooms,
            bathrooms=request.bathrooms,
            sqft=request.sqft,
            zip_code=request.zip_code,
            condition=request.condition
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Home Price Predictor API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
            "openapi": "/openapi.json"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
