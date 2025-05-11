import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def train_weather_model(input_file_path, output_model_path):
    """
    Train a weather prediction model using preprocessed weather data.
    
    Args:
        input_file_path (str): Path to the preprocessed weather data CSV
        output_model_path (str): Path to save the trained model
        
    Returns:
        dict: Dictionary containing model performance metrics
    """
    # Load preprocessed data
    print(f"Loading preprocessed data from {input_file_path}")
    data = pd.read_csv(input_file_path)
    
    # Define features and target
    X = data.drop('Temperature (°C)', axis=1)
    y = data['Temperature (°C)']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Train model
    print("Training RandomForest model...")
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model performance: MSE = {mse:.2f}, R² = {r2:.2f}")
    
    # Save model
    joblib.dump(model, output_model_path)
    print(f"Model saved to {output_model_path}")
    
    return {
        'mse': mse,
        'r2': r2,
        'feature_importance': dict(zip(X.columns, model.feature_importances_))
    }