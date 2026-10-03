"""Weather client for Open-Meteo geocoding and live forecasts.

No LLM logic belongs in this file. All errors are typed and raised cleanly.
"""

from typing import Dict, Any, Optional
import requests
from backend.models import WeatherData

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_FIELDS = "temperature_2m,wind_speed_10m,precipitation,precipitation_probability,uv_index"
DEFAULT_TIMEOUT = 10  # seconds


class WeatherError(Exception):
    """Base exception for weather-related failures."""
    pass


class LocationUnresolved(WeatherError):
    """Raised when geocoding fails to resolve a city name."""
    def __init__(self, location: str):
        super().__init__(f"Could not resolve geographic location for '{location}'.")
        self.location = location


class WeatherAPIError(WeatherError):
    """Raised when the weather forecast API call fails or returns invalid data."""
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error


def geocode_location(location_name: str, timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """Geocode a city name using Open-Meteo geocoding API.

    Returns the first matching result's resolved name, latitude, and longitude.
    Raises LocationUnresolved if no matches are found, or WeatherAPIError on network error.
    """
    cleaned_name = location_name.strip()
    if not cleaned_name:
        raise LocationUnresolved(location_name)

    try:
        response = requests.get(
            GEOCODING_URL,
            params={"name": cleaned_name, "count": 1, "language": "en", "format": "json"},
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        raise WeatherAPIError(f"Geocoding request failed: {e}", original_error=e) from e
    except Exception as e:
        raise WeatherAPIError(f"Failed to parse geocoding response: {e}", original_error=e) from e

    results = data.get("results")
    if not results or len(results) == 0:
        raise LocationUnresolved(cleaned_name)

    first_hit = results[0]
    return {
        "name": first_hit.get("name", cleaned_name),
        "country": first_hit.get("country", ""),
        "latitude": float(first_hit["latitude"]),
        "longitude": float(first_hit["longitude"]),
    }


def fetch_weather(
    latitude: float,
    longitude: float,
    location_name: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> WeatherData:
    """Fetch live weather forecast from Open-Meteo for given coordinates.

    Explicitly requests required current fields and validates with Pydantic.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": WEATHER_FIELDS,
    }

    try:
        response = requests.get(FORECAST_URL, params=params, timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        raise WeatherAPIError(f"Forecast request failed: {e}", original_error=e) from e
    except Exception as e:
        raise WeatherAPIError(f"Failed to parse forecast response: {e}", original_error=e) from e

    current = data.get("current")
    if not current:
        raise WeatherAPIError("Open-Meteo response did not contain 'current' weather block.")

    # Validate that all required numeric fields are present
    required_fields = ["temperature_2m", "wind_speed_10m", "precipitation", "precipitation_probability", "uv_index"]
    for field in required_fields:
        if field not in current or current[field] is None:
            raise WeatherAPIError(f"Forecast missing required field '{field}'.")

    return WeatherData(
        time=str(current.get("time", "")),
        temperature_2m=float(current["temperature_2m"]),
        wind_speed_10m=float(current["wind_speed_10m"]),
        precipitation=float(current["precipitation"]),
        precipitation_probability=float(current["precipitation_probability"]),
        uv_index=float(current["uv_index"]),
        latitude=latitude,
        longitude=longitude,
        location_name=location_name,
    )


def get_weather_for_location(location_name: str, timeout: int = DEFAULT_TIMEOUT) -> WeatherData:
    """End-to-end deterministic weather fetch: Geocode -> Fetch -> WeatherData."""
    geo = geocode_location(location_name, timeout=timeout)
    display_name = f"{geo['name']}, {geo['country']}" if geo.get("country") else geo["name"]
    return fetch_weather(
        latitude=geo["latitude"],
        longitude=geo["longitude"],
        location_name=display_name,
        timeout=timeout,
    )
