"""
IT Risk Classification API
FastAPI application that serves IT risk predictions using a trained classification model.
Classifies cloud infrastructure security risks into Low, Medium, High, and Critical categories.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import os
import glob
import numpy as np
from typing import Optional, Dict, List
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="IT Risk Classification API",
    description="API to classify IT security risks for cloud infrastructure",
    version="1.0.0"
)

# Models directory
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

# Load the latest trained model, scaler, and features
def load_latest_model():
    """Load the most recently trained model"""
    try:
        # Find latest model files
        model_files = sorted(glob.glob(os.path.join(MODELS_DIR, 'it_risk_model_*.joblib')))
        scaler_files = sorted(glob.glob(os.path.join(MODELS_DIR, 'it_risk_scaler_*.joblib')))
        feature_files = sorted(glob.glob(os.path.join(MODELS_DIR, 'it_risk_features_*.joblib')))
        
        if not model_files or not scaler_files or not feature_files:
            raise FileNotFoundError("IT risk model files not found. Train the model first.")
        
        # Load latest files
        model = joblib.load(model_files[-1])
        scaler = joblib.load(scaler_files[-1])
        features = joblib.load(feature_files[-1])
        
        return model, scaler, features, model_files[-1]
    
    except Exception as e:
        raise RuntimeError(f"Failed to load model files: {e}")

# Load model on startup
try:
    model, scaler, features, model_path = load_latest_model()
    model_loaded = True
    model_timestamp = os.path.basename(model_path).split('_')[3:5]
    model_info = f"Model trained at: {model_timestamp[0]}_{model_timestamp[1]}"
except Exception as e:
    model_loaded = False
    model = None
    scaler = None
    features = None
    model_info = str(e)


# Request/Response Models
class ITRiskAssessmentRequest(BaseModel):
    """Request model for IT risk assessment"""
    firewall_enabled: int = Field(..., ge=0, le=1, description="Firewall enabled (0/1)")
    encryption_level: int = Field(..., ge=0, le=3, description="Encryption level (0=None, 1=Basic, 2=Standard, 3=Strong)")
    mfa_enabled: int = Field(..., ge=0, le=1, description="Multi-factor authentication enabled (0/1)")
    compliance_score: int = Field(..., ge=0, le=100, description="Compliance score (0-100)")
    num_security_patches: int = Field(..., ge=0, description="Number of applied security patches")
    unpatched_vulnerabilities: int = Field(..., ge=0, description="Number of unpatched vulnerabilities")
    backup_enabled: int = Field(..., ge=0, le=1, description="Automated backup enabled (0/1)")
    disk_encryption: int = Field(..., ge=0, le=1, description="Disk encryption enabled (0/1)")
    failed_login_attempts: int = Field(..., ge=0, description="Failed login attempts in last 24 hours")
    unusual_traffic: int = Field(..., ge=0, le=1, description="Unusual traffic detected (0/1)")
    privileged_access_changes: int = Field(..., ge=0, description="Number of privileged access changes in last 24 hours")
    security_events_per_day: int = Field(..., ge=0, description="Average security events per day")

    class Config:
        json_schema_extra = {
            "example": {
                "firewall_enabled": 1,
                "encryption_level": 3,
                "mfa_enabled": 1,
                "compliance_score": 95,
                "num_security_patches": 45,
                "unpatched_vulnerabilities": 2,
                "backup_enabled": 1,
                "disk_encryption": 1,
                "failed_login_attempts": 10,
                "unusual_traffic": 0,
                "privileged_access_changes": 2,
                "security_events_per_day": 50
            }
        }


class RiskProbabilities(BaseModel):
    """Risk probabilities for each class"""
    Low: float
    Medium: float
    High: float
    Critical: float


class ITRiskAssessmentResponse(BaseModel):
    """Response model for IT risk assessment"""
    risk_classification: str = Field(..., description="Predicted risk class (Low, Medium, High, Critical)")
    risk_probabilities: RiskProbabilities
    confidence: float = Field(..., ge=0, le=1, description="Confidence in the prediction (0-1)")
    assessment_timestamp: datetime


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_info: Optional[str] = None


class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_loaded: bool
    model_path: Optional[str] = None
    features: Optional[List[str]] = None
    risk_classes: Optional[List[str]] = None
    message: Optional[str] = None


class BatchAssessmentRequest(BaseModel):
    """Request model for batch risk assessment"""
    assessments: List[ITRiskAssessmentRequest]


class BatchAssessmentResponse(BaseModel):
    """Response model for batch risk assessment"""
    assessments: List[ITRiskAssessmentResponse]
    total: int
    timestamp: datetime


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        model_info=model_info
    )


@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the loaded model"""
    if not model_loaded:
        return ModelInfoResponse(
            model_loaded=False,
            message=model_info
        )
    
    return ModelInfoResponse(
        model_loaded=True,
        model_path=model_path,
        features=features,
        risk_classes=list(model.classes_) if hasattr(model, 'classes_') else None,
        message="Model successfully loaded"
    )


@app.post("/assess", response_model=ITRiskAssessmentResponse)
async def assess_risk(request: ITRiskAssessmentRequest):
    """
    Assess IT risk for a cloud infrastructure
    
    This endpoint classifies the security risk level based on various security
    and compliance factors.
    
    Returns:
    - **risk_classification**: The predicted risk level (Low, Medium, High, Critical)
    - **risk_probabilities**: Probability scores for each risk class
    - **confidence**: Confidence score for the prediction
    - **assessment_timestamp**: When the assessment was made
    """
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        # Prepare features in the correct order
        feature_values = np.array([[
            request.firewall_enabled,
            request.encryption_level,
            request.mfa_enabled,
            request.compliance_score,
            request.num_security_patches,
            request.unpatched_vulnerabilities,
            request.backup_enabled,
            request.disk_encryption,
            request.failed_login_attempts,
            request.unusual_traffic,
            request.privileged_access_changes,
            request.security_events_per_day
        ]], dtype=float)
        
        # Scale features
        features_scaled = scaler.transform(feature_values)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Calculate confidence as the max probability
        confidence = float(np.max(probabilities))
        
        # Map probabilities to class names
        prob_dict = {
            model.classes_[i]: float(probabilities[i])
            for i in range(len(model.classes_))
        }
        
        # Ensure all risk classes are present in response
        risk_probs = RiskProbabilities(
            Low=prob_dict.get('Low', 0.0),
            Medium=prob_dict.get('Medium', 0.0),
            High=prob_dict.get('High', 0.0),
            Critical=prob_dict.get('Critical', 0.0)
        )
        
        return ITRiskAssessmentResponse(
            risk_classification=prediction,
            risk_probabilities=risk_probs,
            confidence=confidence,
            assessment_timestamp=datetime.now()
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )


@app.post("/assess-batch", response_model=BatchAssessmentResponse)
async def assess_risk_batch(request: BatchAssessmentRequest):
    """
    Assess IT risk for multiple infrastructures (batch mode)
    
    Useful for assessing multiple cloud resources at once.
    """
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    if not request.assessments:
        raise HTTPException(
            status_code=400,
            detail="At least one assessment is required"
        )
    
    if len(request.assessments) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 assessments per request"
        )
    
    assessments = []
    
    try:
        for idx, item in enumerate(request.assessments):
            # Prepare features
            feature_values = np.array([[
                item.firewall_enabled,
                item.encryption_level,
                item.mfa_enabled,
                item.compliance_score,
                item.num_security_patches,
                item.unpatched_vulnerabilities,
                item.backup_enabled,
                item.disk_encryption,
                item.failed_login_attempts,
                item.unusual_traffic,
                item.privileged_access_changes,
                item.security_events_per_day
            ]], dtype=float)
            
            # Scale features
            features_scaled = scaler.transform(feature_values)
            
            # Make prediction
            prediction = model.predict(features_scaled)[0]
            probabilities = model.predict_proba(features_scaled)[0]
            
            # Calculate confidence
            confidence = float(np.max(probabilities))
            
            # Map probabilities to class names
            prob_dict = {
                model.classes_[i]: float(probabilities[i])
                for i in range(len(model.classes_))
            }
            
            risk_probs = RiskProbabilities(
                Low=prob_dict.get('Low', 0.0),
                Medium=prob_dict.get('Medium', 0.0),
                High=prob_dict.get('High', 0.0),
                Critical=prob_dict.get('Critical', 0.0)
            )
            
            assessment = ITRiskAssessmentResponse(
                risk_classification=prediction,
                risk_probabilities=risk_probs,
                confidence=confidence,
                assessment_timestamp=datetime.now()
            )
            assessments.append(assessment)
        
        return BatchAssessmentResponse(
            assessments=assessments,
            total=len(assessments),
            timestamp=datetime.now()
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing batch assessments: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "IT Risk Classification API",
        "version": "1.0.0",
        "description": "Classifies IT security risks for cloud infrastructure",
        "status": "healthy" if model_loaded else "unhealthy",
        "endpoints": {
            "health": "/health",
            "model_info": "/model-info",
            "assess": "/assess",
            "assess_batch": "/assess-batch",
            "docs": "/docs",
            "openapi": "/openapi.json"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("IT RISK CLASSIFICATION API")
    print("=" * 60)
    print(f"Model Status: {'Loaded' if model_loaded else 'Not Loaded'}")
    print(f"Model Info: {model_info}")
    print("\nStarting server on http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
