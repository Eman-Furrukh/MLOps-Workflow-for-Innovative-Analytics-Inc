    # train_model.py

import unittest
import os
import pandas as pd
from collect_data import fetch_weather, write_to_csv
from preprocess_data import preprocess_weather_data
from train_model import train_weather_model
import joblib

class TestWeatherPipeline(unittest.TestCase):

    # Test fetch_weather function
    def test_fetch_weather(self):
        weather_data = fetch_weather()
        self.assertIsNotNone(weather_data)
        self.assertEqual(len(weather_data), 6)  # Checking if the data has 6 elements (Timestamp, City, Temp, Weather, Humidity, Wind Speed)
        self.assertIsInstance(weather_data[0], str)  # Timestamp is a string
        self.assertIsInstance(weather_data[2], float)  # Temp is a float
        self.assertIsInstance(weather_data[3], str)  # Weather description is a string

    # Test write_to_csv function
    def test_write_to_csv_creates_and_writes(self):
        weather_data = ['2025-05-10 12:00:00', 'Glasgow', 15.5, 'clear sky', 60, 5.0]
        path = os.path.join(os.path.dirname(__file__), 'test_weather_data.csv')
        write_to_csv(weather_data, path)
        self.assertTrue(os.path.exists(path))

        # Check that the file has data written
        df = pd.read_csv(path)
        self.assertGreater(len(df), 0)  # Make sure there is data in the CSV file
        os.remove(path)  # Clean up the test file

    # Test preprocess_weather_data function
    def test_preprocess_weather_data(self):
        raw_csv_path = os.path.join(os.path.dirname(__file__), 'test_weather_data.csv')
        processed_csv_path = os.path.join(os.path.dirname(__file__), 'processed_weather_data.csv')
        write_to_csv(['2025-05-10 12:00:00', 'Glasgow', 15.5, 'clear sky', 60, 5.0], raw_csv_path)

        preprocess_weather_data(raw_csv_path, processed_csv_path)

        # Check if the processed file is created
        self.assertTrue(os.path.exists(processed_csv_path))

        # Check if the preprocessed file has the correct columns
        df = pd.read_csv(processed_csv_path)
        expected_columns = ['Hour', 'Temperature (°C)', 'Humidity (%)', 'Wind Speed (m/s)', 'Is_Rainy']
        for col in expected_columns:
            self.assertIn(col, df.columns)

        os.remove(raw_csv_path)
        os.remove(processed_csv_path)

    # Test train_weather_model function
    def test_train_weather_model(self):
        processed_csv_path = os.path.join(os.path.dirname(__file__), 'processed_weather_data.csv')
        model_path = os.path.join(os.path.dirname(__file__), 'weather_model.pkl')

        # Create a dummy preprocessed CSV for testing
        write_to_csv(['2025-05-10 12:00:00', 'Glasgow', 15.5, 'clear sky', 60, 5.0], processed_csv_path)

        # Train the model
        train_weather_model(processed_csv_path, model_path)

        # Check if the model file exists
        self.assertTrue(os.path.exists(model_path))

        # Test if the model is indeed a trained model
        model = joblib.load(model_path)
        self.assertIsNotNone(model)

        os.remove(processed_csv_path)
        os.remove(model_path)

    # Additional test case: Checking if the model file exists after training
    def test_model_file_exists(self):
        model_path = os.path.join(os.path.dirname(__file__), 'weather_model.pkl')
        self.assertTrue(os.path.exists(model_path), "Model file not found after training.")

if __name__ == "__main__":
    unittest.main()
