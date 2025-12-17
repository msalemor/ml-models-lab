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

### IT Risk Classification

1. **Train the model** (run this first):
```bash
python scripts/it_risk_model.py
```
This will:
- Generate 2000 mock cloud infrastructure risk assessments
- Train a Random Forest classification model (100 trees)
- Export model to `scripts/models/it_risk_model_*.joblib`
- Export scaler to `scripts/models/it_risk_scaler_*.joblib`
- Export features to `scripts/models/it_risk_features_*.joblib`
- Export test data to `scripts/models/it_risk_test_data_*.csv`
- Display model accuracy (85% on test data)
- Show feature importance analysis
- Make example predictions

2. **Start the API server**:
```bash
python scripts/it_risk_api.py
```
The API will be available at `http://localhost:8000`

3. **Access the API**:
- Interactive docs: `http://localhost:8000/docs`
- ReDoc docs: `http://localhost:8000/redoc`
- API root: `http://localhost:8000/`

4. **Test the API** - Single Assessment:
```bash
curl -X POST "http://localhost:8000/assess" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

5. **Test Batch Assessment**:
```bash
curl -X POST "http://localhost:8000/assess-batch" \
  -H "Content-Type: application/json" \
  -d '{
    "assessments": [
      { /* assessment 1 */ },
      { /* assessment 2 */ }
    ]
  }'
```

## Models Directory

The `models/` directory (created automatically) contains:
- `home_price_model.pkl`: Trained home price prediction model
- `home_price_scaler.pkl`: Feature scaler for home price model
- `it_risk_model_*.joblib`: Trained IT risk classification models (timestamped)
- `it_risk_scaler_*.joblib`: Feature scalers for IT risk models (timestamped)
- `it_risk_features_*.joblib`: Feature lists for IT risk models (timestamped)
- `it_risk_test_data_*.csv`: Test data and predictions (timestamped)

## API Endpoints

### Home Price Predictor API (Port 8000)
- `GET /`: API information
- `GET /health`: Health check
- `POST /predict`: Predict home price
- `GET /docs`: Interactive API documentation

### IT Risk Classification API (Port 8000)
- `GET /`: API information
- `GET /health`: Health check
- `GET /model-info`: Get model and feature information
- `POST /assess`: Classify IT risk for single infrastructure
- `POST /assess-batch`: Classify IT risk for multiple infrastructures
- `GET /docs`: Interactive API documentation
- `GET /redoc`: Alternative API documentation

## IT Risk Classification Model

**Features (12 inputs):**
- Security controls: firewall, encryption, MFA, backup, disk encryption
- Compliance: compliance score
- Patch management: patches applied, unpatched vulnerabilities
- Activity monitoring: failed logins, unusual traffic, privileged access changes, security events

**Output (4 classes):**
- Low: Well-secured infrastructure
- Medium: Some security concerns
- High: Significant security gaps
- Critical: Severe vulnerabilities requiring immediate action

**Model Performance:**
- Accuracy: 85% on test data
- Top feature: Compliance Score (15.8% importance)
- Algorithm: Random Forest with 100 trees

## Running Both APIs

To run both services on different ports:

**Terminal 1 - Home Price API (8000):**
```bash
# Default runs on 8000
python scripts/home_price_api.py
```

**Terminal 2 - IT Risk API:**
First, you need to modify the port in it_risk_api.py to 8001, or use environment variables.

Currently, the IT Risk API also runs on port 8000 by default. For development, run them sequentially or modify the port.
