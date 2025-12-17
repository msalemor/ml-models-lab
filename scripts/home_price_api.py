"""
FastAPI service for Home Price Prediction
Loads the trained model and provides prediction endpoints.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os
from typing import Optional

# Initialize FastAPI app
app = FastAPI(
    title="Home Price Predictor API",
    description="Predict home prices based on bedrooms, bathrooms, square footage, and zip code",
    version="1.0.0"
)

# Model paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'home_price_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'home_price_scaler.pkl')

# Global variables for model and scaler
model = None
scaler = None

def load_model():
    """Load the trained model and scaler"""
    global model, scaler
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Please run home_price_model.py first to train and save the model."
        )
    
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Scaler file not found at {SCALER_PATH}. "
            "Please run home_price_model.py first to train and save the scaler."
        )
    
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and scaler loaded successfully!")

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
class HomeFeatures(BaseModel):
    """Input features for home price prediction"""
    bedrooms: int = Field(..., ge=1, le=10, description="Number of bedrooms (1-10)")
    bathrooms: int = Field(..., ge=1, le=10, description="Number of bathrooms (1-10)")
    sqft: int = Field(..., ge=100, le=10000, description="Square footage (100-10000)")
    zip_code: int = Field(..., description="Zip code")
    
    class Config:
        schema_extra = {
            "example": {
                "bedrooms": 3,
                "bathrooms": 2,
                "sqft": 1500,
                "zip_code": 90210
            }
        }

class PricePrediction(BaseModel):
    """Response model for price prediction"""
    predicted_price: float = Field(..., description="Predicted home price in USD")
    bedrooms: int
    bathrooms: int
    sqft: int
    zip_code: int
    
    class Config:
        schema_extra = {
            "example": {
                "predicted_price": 425000.50,
                "bedrooms": 3,
                "bathrooms": 2,
                "sqft": 1500,
                "zip_code": 90210
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
        "message": "Home Price Predictor API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
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

@app.post("/predict", response_model=PricePrediction, tags=["Prediction"])
async def predict_price(features: HomeFeatures):
    """
    Predict home price based on input features
    
    - **bedrooms**: Number of bedrooms
    - **bathrooms**: Number of bathrooms
    - **sqft**: Square footage
    - **zip_code**: Zip code
    """
    global model, scaler
    
    # Load model if not loaded
    if model is None or scaler is None:
        try:
            load_model()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Model not available: {str(e)}"
            )
    
    try:
        # Prepare input data
        input_data = np.array([[
            features.bedrooms,
            features.bathrooms,
            features.sqft,
            features.zip_code
        ]])
        
        # Scale and predict
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        
        return PricePrediction(
            predicted_price=float(prediction),
            bedrooms=features.bedrooms,
            bathrooms=features.bathrooms,
            sqft=features.sqft,
            zip_code=features.zip_code
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
