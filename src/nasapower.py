import openmeteo_requests

import requests_cache
import pandas as pd
from retry_requests import retry
import json

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://api.open-meteo.com/v1/forecast"
params = {
	"latitude": 17.98,
	"longitude": 79.53,
	"current": ["temperature_2m", "relative_humidity_2m", "is_day", "precipitation", "cloud_cover"],
	"timezone": "auto",
	"forecast_days": 1
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation {response.Elevation()} m asl")
print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

# After receiving the response, debug by printing the JSON structure
print(json.dumps(response.__dict__, indent=4))
# Or if using a dict response:
# print(json.dumps(data, indent=4))

# Define data based on the response if needed.
data = response.__dict__

# Now, based on the printed structure, adjust the key access.
# For example, if the key is missing, use a conditional check:
if "properties" in data:
    irradiance = data["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]
else:
    print("The expected key 'properties' was not found in the response.")
    irradiance = None

# Current values. The order of variables needs to be the same as requested.
current = response.Current()

current_temperature_2m = current.Variables(0).Value()

current_relative_humidity_2m = current.Variables(1).Value()

current_is_day = current.Variables(2).Value()

current_precipitation = current.Variables(3).Value()

current_cloud_cover = current.Variables(4).Value()

print(f"Current time {current.Time()}")

print(f"Current temperature_2m {current_temperature_2m}")
print(f"Current relative_humidity_2m {current_relative_humidity_2m}")
print(f"Current is_day {current_is_day}")
print(f"Current precipitation {current_precipitation}")
print(f"Current cloud_cover {current_cloud_cover}")