"""Data models for Weather-Advisory Support Bot."""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class SOPCondition(BaseModel):
    field: str
    operator: str  # gt, gte, lt, lte, eq, in, contains
    value: Any


class SOP(BaseModel):
    id: str
    category: str  # outdoor_exercise, travel, vulnerable_groups, general, situational
    severity: str  # low, moderate, high, critical
    description: str
    conditions: Optional[Dict[str, Any]] = None
    advice: str
    cite_as: str
    situational: bool = False


class UserIntent(BaseModel):
    location: str
    activity: str
    person: Optional[str] = None
    time_window: Optional[str] = None
    original_query: str


class WeatherData(BaseModel):
    time: str
    temperature_2m: float
    wind_speed_10m: float
    precipitation: float
    precipitation_probability: float
    uv_index: float
    latitude: float
    longitude: float
    location_name: str


class GraphState(BaseModel):
    session_id: str = ""
    user_message: str = ""
    intent: Optional[UserIntent] = None
    weather: Optional[WeatherData] = None
    matched_sops: List[SOP] = Field(default_factory=list)
    primary_sop: Optional[SOP] = None
    situational_override: bool = False
    error: Optional[str] = None
    response: str = ""
