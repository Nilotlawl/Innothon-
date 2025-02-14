import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import json
from datetime import datetime
import os
import csv

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Define the API endpoint and parameters
url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": 17.98,
    "longitude": 79.53,
    "current": ["temperature_2m", "relative_humidity_2m", "is_day", "precipitation", "cloud_cover"],
    "timezone": "auto",
    "forecast_days": 1
}

# Request weather data
responses = openmeteo.weather_api(url, params=params)

# Process first location (extend with a loop if you have multiple locations)
response = responses[0]
print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation: {response.Elevation()} m asl")
print(f"Timezone: {response.Timezone()} ({response.TimezoneAbbreviation()})")
print(f"UTC Offset: {response.UtcOffsetSeconds()} s")

# Debug: Print the response JSON structure (if needed)
print(json.dumps(response.__dict__, indent=4))

# Attempt to extract solar irradiance from the response (if available)
data = response.__dict__
if "properties" in data and "parameter" in data["properties"]:
    # Check if the solar irradiance key exists
    if "ALLSKY_SFC_SW_DWN" in data["properties"]["parameter"]:
        irradiance = data["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]
    else:
        print("Solar irradiance key not found.")
        irradiance = None
else:
    print("The expected key 'properties' was not found in the response.")
    irradiance = None

# Get current weather values (the order here should match the order you requested)
current = response.Current()
current_temperature_2m = current.Variables(0).Value()
current_relative_humidity_2m = current.Variables(1).Value()
current_is_day = current.Variables(2).Value()
current_precipitation = current.Variables(3).Value()
current_cloud_cover = current.Variables(4).Value()

# Get the current time from the response (or use system time if necessary)
current_time_str = current.Time()  # Expecting ISO format or similar
try:
    current_time = datetime.fromisoformat(current_time_str)
except Exception:
    # Fallback to current system time if parsing fails
    current_time = datetime.now()

# Extract time-based features
day_of_week = current_time.strftime("%A")
month = current_time.month
time_of_day = current_time.strftime("%H:%M:%S")

# Print the current weather values and time-based features
print(f"Current time: {current_time}")
print(f"Temperature (2m): {current_temperature_2m}°C")
print(f"Relative Humidity (2m): {current_relative_humidity_2m}%")
print(f"Is Day: {current_is_day}")
print(f"Precipitation: {current_precipitation} mm")
print(f"Cloud Cover: {current_cloud_cover}%")
if irradiance is not None:
    print(f"Solar Irradiance: {irradiance}")
print(f"Day of Week: {day_of_week}")
print(f"Month: {month}")
print(f"Time of Day: {time_of_day}")

# Prepare a CSV file path for logging the data
csv_file_path = "data/weather_data.csv"
os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

# Define CSV header (adjust as necessary for your ML features)
csv_header = [
    "timestamp", "temperature_2m", "relative_humidity_2m",
    "is_day", "precipitation", "cloud_cover", "solar_irradiance",
    "day_of_week", "month", "time_of_day"
]

# Check if CSV exists, if not write header
file_exists = os.path.isfile(csv_file_path)
with open(csv_file_path, mode='a', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_header)
    if not file_exists:
        writer.writeheader()

    # Write the current row of data
    writer.writerow({
        "timestamp": current_time.isoformat(),
        "temperature_2m": current_temperature_2m,
        "relative_humidity_2m": current_relative_humidity_2m,
        "is_day": current_is_day,
        "precipitation": current_precipitation,
        "cloud_cover": current_cloud_cover,
        "solar_irradiance": irradiance,
        "day_of_week": day_of_week,
        "month": month,
        "time_of_day": time_of_day
    })

print(f"Data logged to {csv_file_path}")
