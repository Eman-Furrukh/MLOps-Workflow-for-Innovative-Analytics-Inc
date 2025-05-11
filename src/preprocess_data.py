import pandas as pd
import os

# Get the absolute path to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define paths relative to the project root
RAW_CSV = os.path.join(BASE_DIR, "data", "raw", "glasgow_weather_data.csv")
PROCESSED_CSV = os.path.join(BASE_DIR, "data", "processed", "preprocessed_weather_data.csv")


def preprocess_weather_data(input_path, output_path):
    """Preprocess weather data from raw CSV and save cleaned version."""
    # Read data with a specific encoding (ISO-8859-1)
    df = pd.read_csv(input_path, encoding='ISO-8859-1')

    # Drop missing or invalid rows
    df.dropna(inplace=True)

    # Convert 'Timestamp' to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Feature: Hour of the day
    df['Hour'] = df['Timestamp'].dt.hour

    # Feature: Is it rainy/cloudy?
    df['Is_Rainy'] = df['Weather'].str.contains(
        "rain|drizzle|storm|shower", case=False, na=False
    ).astype(int)

    # Drop unneeded columns
    df = df.drop(columns=['City', 'Timestamp'])

    # Reorder columns
    cols = ['Hour', 'Temperature (°C)', 'Humidity (%)', 'Wind Speed (m/s)', 'Is_Rainy']
    df = df[[col for col in cols if col in df.columns]]

    # Save preprocessed file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[✓] Preprocessed data saved to {output_path}")


if __name__ == "__main__":
    preprocess_weather_data(RAW_CSV, PROCESSED_CSV)
