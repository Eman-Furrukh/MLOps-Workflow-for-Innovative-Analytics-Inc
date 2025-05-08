import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
import joblib

# Get the absolute path to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define paths relative to the project root
PROCESSED_CSV = os.path.join(BASE_DIR, "data", "processed", "preprocessed_weather_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "weather_model.pkl")

def train_weather_model(input_path, model_path):
    # Load the preprocessed data
    df = pd.read_csv(input_path)

    # Check if the necessary columns exist in the preprocessed data
    required_columns = ['Hour', 'Humidity (%)', 'Wind Speed (m/s)', 'Is_Rainy']
    for col in required_columns:
        if col not in df.columns:
            print(f"Warning: '{col}' column is missing from the preprocessed data.")

    # Define feature columns (using 'Hour', 'Humidity (%)', 'Wind Speed (m/s)', 'Is_Rainy')
    X = df[['Hour', 'Humidity (%)', 'Wind Speed (m/s)', 'Is_Rainy']]

    # Assuming you're predicting the temperature (or another variable if needed)
    y = df['Humidity (%)']  # Example: predicting humidity, change to your actual target

    # Initialize and train the RandomForestRegressor (or any other model)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save the trained model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"[✓] Model trained and saved to {model_path}")

if __name__ == "__main__":
    train_weather_model(PROCESSED_CSV, MODEL_PATH)
