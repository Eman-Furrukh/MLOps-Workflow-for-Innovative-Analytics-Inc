import requests
import csv
from datetime import datetime
import os
import time

API_KEY = "4d8970db6742d382197749ca3c923bae"
CITY = "Glasgow"
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "glasgow_weather_data.csv")
URL = f"https://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"

def fetch_weather():
    response = requests.get(URL)
    data = response.json()
    if response.status_code == 200:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        temp = data['main']['temp']
        weather = data['weather'][0]['description']
        humidity = data['main']['humidity']
        wind_speed = data['wind']['speed']
        return [timestamp, CITY, temp, weather, humidity, wind_speed]
    return None

def write_to_csv(data, path):
    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    file_exists = os.path.exists(path)
    with open(path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Timestamp', 'City', 'Temp (°C)', 'Weather', 'Humidity (%)', 'Wind Speed (m/s)'])
        writer.writerow(data)


if __name__ == "__main__":
    weather = fetch_weather()
    if weather:
        write_to_csv(weather, OUTPUT_CSV)
        print("Weather data logged.")
#    time.sleep(3600)