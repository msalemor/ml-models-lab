# ML Models Lab - Scripts

This folder contains machine learning models and FastAPI services for various prediction tasks.

## Contents

### 1. Home Price Predictor
- **`home_price_model.py`**: Trains an ML model to predict home prices based on bedrooms, bathrooms, square footage, and zip code
- **`home_price_api.py`**: FastAPI service that loads the model and provides prediction endpoints

### 2. IT Risk Analysis
- **`it_risk_model.py`**: Trains an ML classification model to assess cloud provider risk levels
- **`it_risk_api.py`**: FastAPI service that loads the model and provides risk analysis endpoints

## Setup

1. Install dependencies:
```bash
pip install -r ../requirements.txt
```

## Usage

### Home Price Predictor

1. **Train the model** (run this first):
```bash
python scripts/home_price_model.py
```
This will:
- Generate mock home price data
- Train a Random Forest regression model
- Save the model to `scripts/models/home_price_model.pkl`
- Display sample predictions

2. **Start the API server**:
```bash
python scripts/home_price_api.py
```
The API will be available at `http://localhost:8000`

3. **Test the API**:
- Open `http://localhost:8000/docs` for interactive API documentation
- Or use curl:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "bedrooms": 3,
    "bathrooms": 2,
    "sqft": 1500,
    "zip_code": 90210
  }'
```

### IT Risk Analysis

1. **Train the model** (run this first):
```bash
python scripts/it_risk_model.py
```
This will:
- Generate mock cloud provider risk data
- Train a Random Forest classification model
- Save the model to `scripts/models/it_risk_model.pkl`
- Display model performance metrics

2. **Start the API server**:
```bash
python scripts/it_risk_api.py
```
The API will be available at `http://localhost:8001`

3. **Test the API**:
- Open `http://localhost:8001/docs` for interactive API documentation
- Or use curl:
```bash
curl -X POST "http://localhost:8001/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "AWS",
    "uptime_percentage": 99.9,
    "security_incidents": 2,
    "compliance_score": 95.0,
    "cost_efficiency": 88.5,
    "support_rating": 4.5,
    "data_encryption": 1,
    "backup_frequency_days": 1,
    "response_time_hours": 2.0
  }'
```

## Models Directory

The `models/` directory (created automatically) contains:
- `home_price_model.pkl`: Trained home price prediction model
- `home_price_scaler.pkl`: Feature scaler for home price model
- `it_risk_model.pkl`: Trained IT risk classification model
- `it_risk_scaler.pkl`: Feature scaler for IT risk model
- `it_risk_provider_encoder.pkl`: Label encoder for cloud providers
- `it_risk_encoder.pkl`: Label encoder for risk categories

## API Endpoints

### Home Price Predictor API (Port 8000)
- `GET /`: API information
- `GET /health`: Health check
- `POST /predict`: Predict home price
- `GET /docs`: Interactive API documentation

### IT Risk Analysis API (Port 8001)
- `GET /`: API information
- `GET /health`: Health check
- `POST /analyze`: Analyze cloud provider risk
- `GET /docs`: Interactive API documentation
