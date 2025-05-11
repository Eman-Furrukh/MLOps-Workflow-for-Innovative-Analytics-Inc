import unittest
import os
import pandas as pd
from collect_data import fetch_weather, write_to_csv
from preprocess_data import preprocess_weather_data
from train_model import train_weather_model
import joblib


class TestWeatherPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test files paths."""
        cls.test_dir = os.path.dirname(__file__)
        cls.raw_path = os.path.join(cls.test_dir, 'test_weather_data.csv')
        cls.processed_path = os.path.join(cls.test_dir, 'processed_weather_data.csv')
        cls.model_path = os.path.join(cls.test_dir, 'weather_model.pkl')

    def tearDown(self):
        """Clean up test files."""
        for path in [self.raw_path, self.processed_path, self.model_path]:
            if os.path.exists(path):
                os.remove(path)

    def test_fetch_weather(self):
        """Test fetch_weather function."""
        weather_data = fetch_weather()
        self.assertIsNotNone(weather_data)
        self.assertEqual(len(weather_data), 6)
        self.assertIsInstance(weather_data[0], str)
        self.assertIsInstance(weather_data[2], float)
        self.assertIsInstance(weather_data[3], str)

    def test_write_to_csv_creates_and_writes(self):
        """Test write_to_csv creates and writes data."""
        weather_data = ['2025-05-10 12:00:00', 'Glasgow', 15.5, 'clear sky', 60, 5.0]
        write_to_csv(weather_data, self.raw_path)
        self.assertTrue(os.path.exists(self.raw_path))

        df = pd.read_csv(self.raw_path)
        self.assertGreater(len(df), 0)

    def test_preprocess_weather_data(self):
        """Test preprocessing weather data."""
        write_to_csv(['2025-05-10 12:00:00', 'Glasgow', 15.5, 'clear sky', 60, 5.0], 
                    self.raw_path)

        preprocess_weather_data(self.raw_path, self.processed_path)
        self.assertTrue(os.path.exists(self.processed_path))

        df = pd.read_csv(self.processed_path)
        expected_columns = [
            'Hour', 'Temperature (°C)', 'Humidity (%)', 'Wind Speed (m/s)', 'Is_Rainy'
        ]
        for col in expected_columns:
            self.assertIn(col, df.columns)

    def test_train_weather_model(self):
        """Test model training and file creation."""
        write_to_csv(['2025-05-10 12:00:00', 'Glasgow', 15.5, 'clear sky', 60, 5.0], 
                    self.processed_path)
        train_weather_model(self.processed_path, self.model_path)

        self.assertTrue(os.path.exists(self.model_path))
        model = joblib.load(self.model_path)
        self.assertIsNotNone(model)

    def test_model_file_exists_after_training(self):
        """Ensure model file exists after training."""
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
            
        self.assertFalse(os.path.exists(self.model_path))
        write_to_csv(['2025-05-10 12:00:00', 'Glasgow', 15.5, 'clear sky', 60, 5.0], 
                    self.processed_path)
        train_weather_model(self.processed_path, self.model_path)
        self.assertTrue(os.path.exists(self.model_path))


if __name__ == "__main__":
    unittest.main()