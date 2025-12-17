"""
IT Risk Analysis Model
Trains an ML model to classify the risk level of cloud providers.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# Set random seed for reproducibility
np.random.seed(42)

def generate_mock_data(n_samples=1000):
    """Generate mock IT risk analysis data for cloud providers"""
    
    # Cloud provider types
    providers = ['AWS', 'Azure', 'GCP', 'IBM Cloud', 'Oracle Cloud', 'Other']
    
    # Generate features
    provider = np.random.choice(providers, n_samples)
    uptime_percentage = np.random.uniform(95.0, 99.99, n_samples)
    security_incidents = np.random.randint(0, 20, n_samples)
    compliance_score = np.random.uniform(50, 100, n_samples)
    cost_efficiency = np.random.uniform(60, 100, n_samples)
    support_rating = np.random.uniform(3.0, 5.0, n_samples)
    data_encryption = np.random.choice([0, 1], n_samples, p=[0.1, 0.9])  # 90% have encryption
    backup_frequency = np.random.choice([1, 7, 14, 30], n_samples)  # days
    response_time_hours = np.random.uniform(0.5, 48, n_samples)
    
    # Calculate risk based on features
    risk_scores = []
    
    for i in range(n_samples):
        score = 0
        
        # Uptime contribution (lower uptime = higher risk)
        if uptime_percentage[i] < 98:
            score += 30
        elif uptime_percentage[i] < 99:
            score += 20
        elif uptime_percentage[i] < 99.5:
            score += 10
        
        # Security incidents (more incidents = higher risk)
        if security_incidents[i] > 10:
            score += 25
        elif security_incidents[i] > 5:
            score += 15
        elif security_incidents[i] > 2:
            score += 5
        
        # Compliance score (lower score = higher risk)
        if compliance_score[i] < 70:
            score += 25
        elif compliance_score[i] < 85:
            score += 10
        
        # Encryption (no encryption = higher risk)
        if data_encryption[i] == 0:
            score += 20
        
        # Backup frequency (less frequent = higher risk)
        if backup_frequency[i] > 14:
            score += 15
        elif backup_frequency[i] > 7:
            score += 8
        
        # Response time (slower = higher risk)
        if response_time_hours[i] > 24:
            score += 15
        elif response_time_hours[i] > 12:
            score += 8
        elif response_time_hours[i] > 4:
            score += 3
        
        # Support rating (lower = higher risk)
        if support_rating[i] < 3.5:
            score += 10
        elif support_rating[i] < 4.0:
            score += 5
        
        risk_scores.append(score)
    
    # Convert scores to risk categories
    risk_category = []
    for score in risk_scores:
        if score < 30:
            risk_category.append('Low')
        elif score < 60:
            risk_category.append('Medium')
        else:
            risk_category.append('High')
    
    # Create DataFrame
    df = pd.DataFrame({
        'provider': provider,
        'uptime_percentage': uptime_percentage,
        'security_incidents': security_incidents,
        'compliance_score': compliance_score,
        'cost_efficiency': cost_efficiency,
        'support_rating': support_rating,
        'data_encryption': data_encryption,
        'backup_frequency_days': backup_frequency,
        'response_time_hours': response_time_hours,
        'risk_category': risk_category
    })
    
    return df

def train_model():
    """Train the IT risk analysis model"""
    
    print("Generating mock data...")
    df = generate_mock_data(1000)
    
    # Display sample data
    print("\nSample data:")
    print(df.head(10))
    print(f"\nDataset shape: {df.shape}")
    print(f"\nRisk category distribution:")
    print(df['risk_category'].value_counts())
    
    # Encode categorical variables
    le_provider = LabelEncoder()
    df['provider_encoded'] = le_provider.fit_transform(df['provider'])
    
    le_risk = LabelEncoder()
    df['risk_encoded'] = le_risk.fit_transform(df['risk_category'])
    
    # Prepare features and target
    feature_columns = [
        'provider_encoded', 'uptime_percentage', 'security_incidents',
        'compliance_score', 'cost_efficiency', 'support_rating',
        'data_encryption', 'backup_frequency_days', 'response_time_hours'
    ]
    
    X = df[feature_columns]
    y = df['risk_encoded']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("\nTraining model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    print(f"\nModel Performance:")
    print(f"Training Accuracy: {train_score:.4f}")
    print(f"Testing Accuracy: {test_score:.4f}")
    
    # Classification report
    y_pred = model.predict(X_test_scaled)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le_risk.classes_))
    
    # Feature importance
    print("\nFeature Importances:")
    feature_names = feature_columns
    importances = model.feature_importances_
    for name, importance in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
        print(f"{name}: {importance:.4f}")
    
    # Save model, scaler, and encoders
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'it_risk_model.pkl')
    scaler_path = os.path.join(model_dir, 'it_risk_scaler.pkl')
    provider_encoder_path = os.path.join(model_dir, 'it_risk_provider_encoder.pkl')
    risk_encoder_path = os.path.join(model_dir, 'it_risk_encoder.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(le_provider, provider_encoder_path)
    joblib.dump(le_risk, risk_encoder_path)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"Provider encoder saved to: {provider_encoder_path}")
    print(f"Risk encoder saved to: {risk_encoder_path}")
    
    # Test predictions
    print("\nTest predictions on sample data:")
    sample_data = pd.DataFrame({
        'provider': ['AWS', 'Azure', 'Other'],
        'uptime_percentage': [99.9, 98.5, 96.0],
        'security_incidents': [1, 5, 15],
        'compliance_score': [95, 85, 65],
        'cost_efficiency': [90, 80, 70],
        'support_rating': [4.5, 4.0, 3.2],
        'data_encryption': [1, 1, 0],
        'backup_frequency_days': [1, 7, 30],
        'response_time_hours': [2, 8, 36]
    })
    
    sample_data['provider_encoded'] = le_provider.transform(sample_data['provider'])
    X_sample = sample_data[feature_columns]
    X_sample_scaled = scaler.transform(X_sample)
    predictions = model.predict(X_sample_scaled)
    predicted_risks = le_risk.inverse_transform(predictions)
    
    for i, (provider, risk) in enumerate(zip(sample_data['provider'], predicted_risks)):
        print(f"Provider {i+1} ({provider}): Risk = {risk}")
    
    return model, scaler, le_provider, le_risk

if __name__ == "__main__":
    train_model()
