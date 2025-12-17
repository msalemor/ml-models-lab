"""
Home Price Predictor Model
Trains an ML model to predict home prices based on bedrooms, baths, square footage, zip code, and condition.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

# Set random seed for reproducibility
np.random.seed(42)

def generate_mock_data(n_samples=1000):
    """Generate mock home price data"""
    
    # Generate features
    bedrooms = np.random.randint(1, 6, n_samples)  # 1-5 bedrooms
    bathrooms = np.random.randint(1, 4, n_samples)  # 1-3 bathrooms
    sqft = np.random.randint(800, 4000, n_samples)  # 800-4000 sqft
    zip_codes = np.random.choice([90001, 90210, 10001, 10011, 60601, 60611, 94102, 94110], n_samples)
    conditions = np.random.choice(['Excellent', 'Good', 'Fair', 'Remodel'], n_samples)
    
    # Create base price with some logic
    base_price = (
        sqft * 150 +  # $150 per sqft
        bedrooms * 50000 +  # $50k per bedroom
        bathrooms * 30000  # $30k per bathroom
    )
    
    # Zip code premium/discount
    zip_premium = {
        90001: 0.8, 90210: 1.5, 10001: 1.3, 10011: 1.25,
        60601: 1.0, 60611: 1.1, 94102: 1.4, 94110: 1.35
    }
    
    # Condition multiplier
    condition_multiplier = {
        'Excellent': 1.2,
        'Good': 1.0,
        'Fair': 0.85,
        'Remodel': 0.7
    }
    
    price = np.array([
        base_price[i] * zip_premium[zip_codes[i]] * condition_multiplier[conditions[i]]
        for i in range(n_samples)
    ])
    
    # Add random noise
    noise = np.random.normal(0, 50000, n_samples)
    price = np.maximum(price + noise, 100000)
    
    df = pd.DataFrame({
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'sqft': sqft,
        'zip_code': zip_codes,
        'condition': conditions,
        'price': price
    })
    
    return df

def train_model():
    """Train the home price prediction model"""
    
    print("Generating mock data...")
    df = generate_mock_data(1000)
    
    print("\nSample data:")
    print(df.head(10))
    print(f"\nDataset shape: {df.shape}")
    print(f"\nPrice statistics:\n{df['price'].describe()}")
    
    # Prepare features
    X = df[['bedrooms', 'bathrooms', 'sqft', 'zip_code', 'condition']].copy()
    y = df['price']
    
    # Encode condition
    condition_encoder = LabelEncoder()
    X['condition'] = condition_encoder.fit_transform(X['condition'])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("\nTraining model...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    print(f"\nModel Performance:")
    print(f"Training R² Score: {model.score(X_train_scaled, y_train):.4f}")
    print(f"Testing R² Score: {model.score(X_test_scaled, y_test):.4f}")
    
    # Feature importance
    feature_names = ['bedrooms', 'bathrooms', 'sqft', 'zip_code', 'condition']
    print("\nFeature Importances:")
    for name, importance in zip(feature_names, model.feature_importances_):
        print(f"{name}: {importance:.4f}")
    
    # Save model, scaler, and encoder
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(model, os.path.join(model_dir, 'home_price_model.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'home_price_scaler.pkl'))
    joblib.dump(condition_encoder, os.path.join(model_dir, 'condition_encoder.pkl'))
    
    print(f"\nModel saved to: {model_dir}")
    
    # Test predictions
    print("\nTest predictions on sample data:")
    sample_data = np.array([
        [3, 2, 1500, 90210, 'Excellent'],
        [2, 1, 1000, 90001, 'Fair'],
        [4, 3, 2500, 10001, 'Good'],
    ])
    
    sample_encoded = sample_data.copy()
    sample_encoded[:, 4] = condition_encoder.transform(sample_data[:, 4])
    sample_scaled = scaler.transform(sample_encoded.astype(float))
    predictions = model.predict(sample_scaled)
    
    for data, pred in zip(sample_data, predictions):
        print(f"{data[0]} bed, {data[1]} bath, {data[2]} sqft, zip {data[3]}, {data[4]} -> ${pred:,.2f}")
    
    return model, scaler, condition_encoder

if __name__ == "__main__":
    train_model()
