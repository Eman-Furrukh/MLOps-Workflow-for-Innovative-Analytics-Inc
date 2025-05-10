import unittest
import os
import pandas as pd
import numpy as np
import joblib
from tempfile import TemporaryDirectory

# Import the train_weather_model function
# Adjust the import path based on your project structure
from train_model import train_weather_model

class TestTrainWeatherModel(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = TemporaryDirectory()
        self.input_csv = os.path.join(self.test_dir.name, "test_data.csv")
        self.model_path = os.path.join(self.test_dir.name, "test_model.pkl")

        # Create a sample DataFrame
        data = {
            'Hour': np.random.randint(0, 24, size=10),
            'Humidity (%)': np.random.uniform(30, 90, size=10),
            'Wind Speed (m/s)': np.random.uniform(0, 10, size=10),
            'Is_Rainy': np.random.randint(0, 2, size=10)
        }
        df = pd.DataFrame(data)
        df.to_csv(self.input_csv, index=False)

    def test_train_weather_model(self):
        # Call the function to train the model
        train_weather_model(self.input_csv, self.model_path)

        # Check if the model file was created
        self.assertTrue(os.path.exists(self.model_path), "Model file was not created.")

        # Load the model and check if it's a RandomForestRegressor instance
        model = joblib.load(self.model_path)
        from sklearn.ensemble import RandomForestRegressor
        self.assertIsInstance(model, RandomForestRegressor, "Loaded model is not a RandomForestRegressor.")

    def tearDown(self):
        # Clean up the temporary directory
        self.test_dir.cleanup()

if __name__ == '__main__':
    unittest.main()
