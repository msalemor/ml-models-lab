"""
Home Price Predictor Model
Trains an ML model to predict home prices based on bedrooms, baths, square footage, and zip code.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
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
    
    # Create base price with some logic
    # Base price influenced by sqft, bedrooms, bathrooms
    base_price = (
        sqft * 150 +  # $150 per sqft
        bedrooms * 50000 +  # $50k per bedroom
        bathrooms * 30000  # $30k per bathroom
    )
    
    # Zip code premium/discount
    zip_premium = {
        90001: 0.8,   # Lower cost area
        90210: 1.5,   # Beverly Hills premium
        10001: 1.3,   # Manhattan premium
        10011: 1.25,  # Manhattan premium
        60601: 1.0,   # Chicago standard
        60611: 1.1,   # Chicago premium
        94102: 1.4,   # San Francisco premium
        94110: 1.35   # San Francisco premium
    }
    
    price = np.array([base_price[i] * zip_premium[zip_codes[i]] for i in range(n_samples)])
    
    # Add some random noise
    noise = np.random.normal(0, 50000, n_samples)
    price = price + noise
    
    # Ensure positive prices
    price = np.maximum(price, 100000)
    
    # Create DataFrame
    df = pd.DataFrame({
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'sqft': sqft,
        'zip_code': zip_codes,
        'price': price
    })
    
    return df

def train_model():
    """Train the home price prediction model"""
    
    print("Generating mock data...")
    df = generate_mock_data(1000)
    
    # Display sample data
    print("\nSample data:")
    print(df.head(10))
    print(f"\nDataset shape: {df.shape}")
    print(f"\nPrice statistics:")
    print(df['price'].describe())
    
    # Prepare features and target
    X = df[['bedrooms', 'bathrooms', 'sqft', 'zip_code']]
    y = df['price']
    
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
    
    # Evaluate model
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    print(f"\nModel Performance:")
    print(f"Training R² Score: {train_score:.4f}")
    print(f"Testing R² Score: {test_score:.4f}")
    
    # Feature importance
    feature_names = X.columns
    importances = model.feature_importances_
    print("\nFeature Importances:")
    for name, importance in zip(feature_names, importances):
        print(f"{name}: {importance:.4f}")
    
    # Save model and scaler
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'home_price_model.pkl')
    scaler_path = os.path.join(model_dir, 'home_price_scaler.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    
    # Test predictions
    print("\nTest predictions on sample data:")
    sample_data = np.array([
        [3, 2, 1500, 90210],  # 3 bed, 2 bath, 1500 sqft, Beverly Hills
        [2, 1, 1000, 90001],  # 2 bed, 1 bath, 1000 sqft, Lower cost area
        [4, 3, 2500, 10001],  # 4 bed, 3 bath, 2500 sqft, Manhattan
    ])
    
    sample_scaled = scaler.transform(sample_data)
    predictions = model.predict(sample_scaled)
    
    for i, (data, pred) in enumerate(zip(sample_data, predictions)):
        print(f"House {i+1}: {data[0]} bed, {data[1]} bath, {data[2]} sqft, zip {data[3]} -> ${pred:,.2f}")
    
    return model, scaler

if __name__ == "__main__":
    train_model()
