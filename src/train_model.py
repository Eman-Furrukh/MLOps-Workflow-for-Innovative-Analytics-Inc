import unittest
import os
import pandas as pd
from collect_data import fetch_weather, write_to_csv
from preprocess_data import preprocess_weather_data
from train_model import train_weather_model
import joblib


class TestWeatherPipeline(unittest.TestCase):

    def test_fetch_weather(self):
        """Test fetch_weather function"""
        weather_data = fetch_weather()
        self.assertIsNotNone(weather_data)
        self.assertEqual(len(weather_data), 6)  # Timestamp, City, Temp, Weather, Humidity, Wind Speed
        self.assertIsInstance(weather_data[0], str)
        self.assertIsInstance(weather_data[2], float)
        self.assertIsInstance(weather_data[3], str)

    def test_write_to_csv_creates_and_writes(self):
        """Test write_to_csv creates and writes data"""
        weather_data = ['2025-05-10 12:00:00', 'Glasgow', 15.5, 'clear sky', 60, 5.0]
        path = os.path.join(os.path.dirname(__file__), 'test_weather_data.csv')
        write_to_csv(weather_data, path)
        self.assertTrue(os.path.exists(path))

        df = pd.read_csv(path)
        self.assertGreater(len(df), 0)
        os.remove(path)

    def test_preprocess_weather_data(self):
        """Test preprocessing weather data"""
        raw_path = os.path.join(os.path.dirname(__file__), 'test_weather_data.csv')
        processed_path = os.path.join(os.path.dirname(__file__), 'processed_weather_data.csv')
        write_to_csv(['2025-05-10 12:00:00', 'Glasgow', 15.5, 'clear sky', 60, 5.0], raw_path)

        preprocess_weather_data(raw_path, processed_path)
        self.assertTrue(os.path.exists(processed_path))

        df = pd.read_csv(processed_path)
        expected_columns = [
            'Hour', 'Temperature (°C)', 'Humidity (%)', 'Wind Speed (m/s)', 'Is_Rainy'
        ]
        for col in expected_columns:
            self.assertIn(col, df.columns)

        os.remove(raw_path)
        os.remove(processed_path)

    def test_train_weather_model(self):
        """Test model training and file creation"""
        processed_path = os.path.join(os.path.dirname(__file__), 'processed_weather_data.csv')
        model_path = os.path.join(os.path.dirname(__file__), 'weather_model.pkl')

        write_to_csv(['2025-05-10 12:00:00', 'Glasgow', 15.5, 'clear sky', 60, 5.0], processed_path)
        train_weather_model(processed_path, model_path)

        self.assertTrue(os.path.exists(model_path))
        model = joblib.load(model_path)
        self.assertIsNotNone(model)

        os.remove(processed_path)
        os.remove(model_path)

    def test_model_file_exists(self):
        """Ensure model file exists after training"""
        model_path = os.path.join(os.path.dirname(__file__), 'weather_model.pkl')
        self.assertTrue(os.path.exists(model_path), "Model file not found after training.")


if __name__ == "__main__":
    unittest.main()
