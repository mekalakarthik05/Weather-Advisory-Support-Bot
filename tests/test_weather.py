"""Unit tests for Open-Meteo weather client and error handling."""

from unittest.mock import patch, MagicMock
import pytest
import requests
from backend.weather import (
    geocode_location,
    fetch_weather,
    get_weather_for_location,
    LocationUnresolved,
    WeatherAPIError,
)
from backend.models import WeatherData


def test_geocode_success():
    """Verify standard geocoding returns coordinates."""
    res = geocode_location("London")
    assert "latitude" in res
    assert "longitude" in res
    assert "London" in res["name"]


def test_geocode_unresolved_location():
    """Verify that fictional/unresolvable location raises LocationUnresolved."""
    with pytest.raises(LocationUnresolved):
        geocode_location("DefinitelyNotARealCityName998877")


def test_geocode_network_failure():
    """Verify that network exceptions are converted into WeatherAPIError."""
    with patch("requests.get", side_effect=requests.RequestException("Connection refused")):
        with pytest.raises(WeatherAPIError):
            geocode_location("Hyderabad")


def test_fetch_weather_success():
    """Verify live forecast retrieval with explicit fields."""
    # Hyderabad coordinates
    w = fetch_weather(17.385, 78.486, "Hyderabad, India")
    assert isinstance(w, WeatherData)
    assert isinstance(w.temperature_2m, float)
    assert isinstance(w.wind_speed_10m, float)
    assert isinstance(w.precipitation_probability, float)
    assert isinstance(w.uv_index, float)


def test_fetch_weather_missing_fields_failure():
    """Verify that incomplete weather response raises WeatherAPIError."""
    mock_resp = MagicMock()
    # Missing required field 'temperature_2m'
    mock_resp.json.return_value = {
        "current": {
            "time": "2026-10-01T12:00",
            "wind_speed_10m": 12.0,
            "precipitation": 0.0,
        }
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        with pytest.raises(WeatherAPIError) as exc_info:
            fetch_weather(10.0, 20.0, "Test Location")
        assert "missing required field" in str(exc_info.value)
