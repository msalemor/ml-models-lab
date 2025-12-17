"""
IT Risk Classification Model
Classifies cloud infrastructure security risks into Low, Medium, High, and Critical categories
based on various security and compliance factors.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

def generate_mock_data(n_samples=1000):
    """Generate mock IT risk assessment data for cloud providers"""
    
    # Security and compliance factors
    firewall_enabled = np.random.choice([0, 1], n_samples, p=[0.2, 0.8])  # 80% have firewall
    encryption_level = np.random.choice(['None', 'Basic', 'Standard', 'Strong'], n_samples, 
                                       p=[0.1, 0.15, 0.4, 0.35])
    mfa_enabled = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    compliance_score = np.random.randint(0, 101, n_samples)  # 0-100
    
    # Infrastructure factors
    num_security_patches = np.random.randint(0, 50, n_samples)
    unpatched_vulnerabilities = np.random.randint(0, 100, n_samples)
    backup_enabled = np.random.choice([0, 1], n_samples, p=[0.25, 0.75])
    disk_encryption = np.random.choice([0, 1], n_samples, p=[0.2, 0.8])
    
    # Activity factors
    failed_login_attempts = np.random.randint(0, 1000, n_samples)
    unusual_traffic = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    privileged_access_changes = np.random.randint(0, 50, n_samples)
    security_events_per_day = np.random.randint(0, 500, n_samples)
    
    # Encode encryption level
    encryption_mapping = {'None': 0, 'Basic': 1, 'Standard': 2, 'Strong': 3}
    encryption_encoded = [encryption_mapping[e] for e in encryption_level]
    
    # Determine risk level based on features
    risk_scores = (
        (1 - firewall_enabled) * 25 +
        (4 - np.array(encryption_encoded)) * 10 +
        (1 - mfa_enabled) * 20 +
        (100 - compliance_score) * 0.3 +
        np.clip(unpatched_vulnerabilities / 10, 0, 30) +
        (1 - backup_enabled) * 15 +
        (1 - disk_encryption) * 15 +
        np.clip(failed_login_attempts / 50, 0, 20) +
        unusual_traffic * 15 +
        np.clip(privileged_access_changes / 5, 0, 20) +
        np.clip(security_events_per_day / 50, 0, 20)
    )
    
    # Add some noise
    risk_scores += np.random.normal(0, 5, n_samples)
    risk_scores = np.clip(risk_scores, 0, 100)
    
    # Classify risks
    def classify_risk(score):
        if score < 25:
            return 'Low'
        elif score < 50:
            return 'Medium'
        elif score < 75:
            return 'High'
        else:
            return 'Critical'
    
    risk_class = [classify_risk(score) for score in risk_scores]
    
    df = pd.DataFrame({
        'firewall_enabled': firewall_enabled,
        'encryption_level': encryption_encoded,
        'mfa_enabled': mfa_enabled,
        'compliance_score': compliance_score,
        'num_security_patches': num_security_patches,
        'unpatched_vulnerabilities': unpatched_vulnerabilities,
        'backup_enabled': backup_enabled,
        'disk_encryption': disk_encryption,
        'failed_login_attempts': failed_login_attempts,
        'unusual_traffic': unusual_traffic,
        'privileged_access_changes': privileged_access_changes,
        'security_events_per_day': security_events_per_day,
        'risk_score': risk_scores,
        'risk_class': risk_class
    })
    
    return df

def train_model():
    """Train the IT risk classification model"""
    
    print("=" * 60)
    print("IT RISK CLASSIFICATION MODEL TRAINING")
    print("=" * 60)
    
    print("\nGenerating mock IT risk assessment data...")
    df = generate_mock_data(2000)
    
    print("\nSample data (first 10 rows):")
    print(df.head(10))
    
    print(f"\nDataset shape: {df.shape}")
    
    print("\nRisk class distribution:")
    print(df['risk_class'].value_counts().sort_index())
    
    print("\nRisk score statistics:")
    print(df['risk_score'].describe())
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col not in ['risk_score', 'risk_class']]
    X = df[feature_cols].copy()
    y = df['risk_class']
    
    print(f"\nFeatures used: {feature_cols}")
    
    # Split data (without stratify to avoid issues with imbalanced classes)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train classifier
    print("\nTraining Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate on test set
    print("\nEvaluating model...")
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=['Low', 'Medium', 'High', 'Critical'])
    print(cm)
    
    # Feature importance
    print("\nTop 10 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(10).to_string(index=False))
    
    # Create models directory if it doesn't exist
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model and scaler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(model_dir, f'it_risk_model_{timestamp}.joblib')
    scaler_path = os.path.join(model_dir, f'it_risk_scaler_{timestamp}.joblib')
    feature_cols_path = os.path.join(model_dir, f'it_risk_features_{timestamp}.joblib')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_cols, feature_cols_path)
    
    print(f"\n{'=' * 60}")
    print("MODEL EXPORTED SUCCESSFULLY")
    print(f"{'=' * 60}")
    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"Features saved to: {feature_cols_path}")
    
    # Also save test data for reference
    test_data_path = os.path.join(model_dir, f'it_risk_test_data_{timestamp}.csv')
    test_df = X_test.copy()
    test_df['actual_risk'] = y_test.values
    test_df['predicted_risk'] = y_pred
    test_df.to_csv(test_data_path, index=False)
    print(f"Test data saved to: {test_data_path}")
    
    print(f"\nModel training completed!")
    print(f"Files location: {model_dir}")
    
    return model, scaler, feature_cols

def predict_risk(model, scaler, features_list, input_data):
    """
    Predict IT risk for new data
    
    Args:
        model: Trained classifier
        scaler: Fitted scaler
        features_list: List of feature names in correct order
        input_data: Dictionary or list of values for prediction
    
    Returns:
        Risk class prediction and probability scores
    """
    # Convert input to proper format
    if isinstance(input_data, dict):
        X = np.array([input_data[f] for f in features_list]).reshape(1, -1)
    else:
        X = np.array(input_data).reshape(1, -1)
    
    # Scale input
    X_scaled = scaler.transform(X)
    
    # Predict
    prediction = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]
    
    return prediction, probabilities, model.classes_

if __name__ == "__main__":
    model, scaler, feature_cols = train_model()
    
    # Example prediction
    print("\n" + "=" * 60)
    print("EXAMPLE PREDICTION")
    print("=" * 60)
    
    # Create a sample infrastructure with good security
    sample_good = {
        'firewall_enabled': 1,
        'encryption_level': 3,  # Strong
        'mfa_enabled': 1,
        'compliance_score': 95,
        'num_security_patches': 45,
        'unpatched_vulnerabilities': 2,
        'backup_enabled': 1,
        'disk_encryption': 1,
        'failed_login_attempts': 10,
        'unusual_traffic': 0,
        'privileged_access_changes': 2,
        'security_events_per_day': 50
    }
    
    # Create a sample infrastructure with poor security
    sample_poor = {
        'firewall_enabled': 0,
        'encryption_level': 0,  # None
        'mfa_enabled': 0,
        'compliance_score': 20,
        'num_security_patches': 5,
        'unpatched_vulnerabilities': 85,
        'backup_enabled': 0,
        'disk_encryption': 0,
        'failed_login_attempts': 500,
        'unusual_traffic': 1,
        'privileged_access_changes': 40,
        'security_events_per_day': 450
    }
    
    for label, sample in [("Well-Secured Infrastructure", sample_good), 
                          ("Poorly-Secured Infrastructure", sample_poor)]:
        risk_class, probs, classes = predict_risk(model, scaler, feature_cols, sample)
        print(f"\n{label}:")
        print(f"  Predicted Risk Class: {risk_class}")
        print(f"  Risk Probabilities:")
        for cls, prob in zip(classes, probs):
            print(f"    {cls}: {prob:.4f}")
