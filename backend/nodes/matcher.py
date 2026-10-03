"""SOP Matcher and Conflict Resolution Node.

Grounds all safety matching against loaded SOP definitions and live weather data.
Strictly validates SOP IDs against the loaded set and enforces deterministic conflict rules.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from backend.models import SOP, UserIntent, WeatherData

SEVERITY_WEIGHTS = {
    "critical": 4,
    "high": 3,
    "moderate": 2,
    "low": 1,
}


def _evaluate_single_condition(field_value: Any, operator: str, target_value: Any) -> bool:
    """Evaluate a single condition clause deterministically."""
    if field_value is None:
        return False

    op = operator.lower()
    try:
        if op == "gt":
            return float(field_value) > float(target_value)
        elif op == "gte":
            return float(field_value) >= float(target_value)
        elif op == "lt":
            return float(field_value) < float(target_value)
        elif op == "lte":
            return float(field_value) <= float(target_value)
        elif op == "eq":
            return field_value == target_value
        elif op == "in":
            # Target value is a collection (list/set)
            if isinstance(target_value, list):
                # Normalize strings if applicable
                field_str = str(field_value).lower()
                target_strs = [str(x).lower() for x in target_value]
                return any(t in field_str or field_str in t for t in target_strs)
            return field_value in target_value
        elif op == "contains":
            return str(target_value).lower() in str(field_value).lower()
    except (ValueError, TypeError):
        return False

    return False


def _check_conditions_block(
    conditions: Dict[str, Any],
    weather_dict: Dict[str, Any],
    intent_dict: Dict[str, Any],
) -> bool:
    """Evaluate 'all', 'any', and composite condition blocks."""
    # Combine weather and intent fields
    context = {**weather_dict, **intent_dict}

    # Evaluate 'all' list
    if "all" in conditions:
        clause_list = conditions["all"]
        for clause in clause_list:
            field = clause["field"]
            val = context.get(field)
            if not _evaluate_single_condition(val, clause["operator"], clause["value"]):
                return False

    # Evaluate 'any' list
    if "any" in conditions:
        clause_list = conditions["any"]
        any_passed = False
        for clause in clause_list:
            field = clause["field"]
            val = context.get(field)
            if _evaluate_single_condition(val, clause["operator"], clause["value"]):
                any_passed = True
                break
        if not any_passed:
            return False

    # Evaluate fuzzy picnic composite profile if defined
    if "profile" in conditions:
        profile = conditions["profile"]
        activity = str(intent_dict.get("activity", "")).lower()
        if "picnic" not in activity and "outing" not in activity:
            return False

        temp = weather_dict.get("temperature_2m", 0.0)
        wind = weather_dict.get("wind_speed_10m", 0.0)
        precip_prob = weather_dict.get("precipitation_probability", 0.0)
        precip = weather_dict.get("precipitation", 0.0)
        uv = weather_dict.get("uv_index", 0.0)

        temp_ok = profile["ideal_temp_min"] <= temp <= profile["ideal_temp_max"]
        wind_ok = wind <= profile["max_wind"]
        precip_ok = precip_prob <= profile["max_precip_prob"] and precip <= 1.0
        uv_ok = uv <= profile["max_uv"]

        if not (temp_ok and wind_ok and precip_ok and uv_ok):
            return False

    return True


def is_sop_applicable(sop: SOP, intent: UserIntent, weather: WeatherData) -> bool:
    """Determine whether an SOP's conditions are satisfied by live weather and user intent."""
    weather_dict = {
        "temperature_2m": weather.temperature_2m,
        "wind_speed_10m": weather.wind_speed_10m,
        "precipitation": weather.precipitation,
        "precipitation_probability": weather.precipitation_probability,
        "uv_index": weather.uv_index,
    }
    intent_dict = {
        "activity": intent.activity.lower(),
        "person": intent.person.lower() if intent.person else None,
        "location": intent.location,
    }

    # Situational SOPs trigger on severe weather regardless of specific activity
    if sop.situational:
        if sop.conditions:
            return _check_conditions_block(sop.conditions, weather_dict, intent_dict)
        return False

    # General category or activity-specific category check
    # Check if category matches activity
    activity_terms = intent.activity.lower()

    if sop.category == "outdoor_exercise":
        exercise_terms = ["cycle", "cycling", "bike", "run", "running", "jog", "exercise", "workout", "two_wheeler"]
        if not any(term in activity_terms for term in exercise_terms):
            return False

    elif sop.category == "travel":
        travel_terms = ["travel", "commute", "drive", "walking commute", "walk", "two_wheeler", "scooter", "ride"]
        if not any(term in activity_terms for term in travel_terms):
            return False

    elif sop.category == "vulnerable_groups":
        # Requires matching vulnerable person or specific activity
        if not intent.person and not any(term in activity_terms for term in ["kid", "child", "park", "senior", "pet"]):
            return False

    elif sop.category == "general":
        if "picnic" not in activity_terms and "outing" not in activity_terms:
            return False

    # Evaluate numeric and logical conditions
    if sop.conditions:
        return _check_conditions_block(sop.conditions, weather_dict, intent_dict)

    return False


def match_sops(
    intent: UserIntent,
    weather: WeatherData,
    loaded_sops: List[SOP],
    llm_candidate_ids: Optional[List[str]] = None,
) -> Tuple[List[SOP], Optional[SOP], bool]:
    """Match loaded SOPs against intent and live weather.

    Applies deterministic validation, removes any unknown/hallucinated IDs,
    and applies the documented conflict resolution rule:
    1. If a situational SOP applies -> use it as primary policy (situational_override = True).
    2. Otherwise -> choose highest-severity matching SOP as primary.
    3. Multiple matches with same severity are all surfaced in matched_sops.

    Returns: (matched_sops, primary_sop, situational_override)
    """
    valid_id_map: Dict[str, SOP] = {sop.id: sop for sop in loaded_sops}

    # Deterministic matching against all loaded SOPs
    applicable_sops: List[SOP] = []
    for sop in loaded_sops:
        if is_sop_applicable(sop, intent, weather):
            applicable_sops.append(sop)

    # If LLM proposed candidate IDs, strictly validate they exist in loaded set
    if llm_candidate_ids:
        verified_candidates: Set[str] = {
            cid for cid in llm_candidate_ids if cid in valid_id_map
        }
        # Filter applicable to those verified or keep verified applicable
        applicable_sops = [s for s in applicable_sops if s.id in verified_candidates or not verified_candidates]

    if not applicable_sops:
        return [], None, False

    # Check for situational override
    situational_matches = [s for s in applicable_sops if s.situational]
    if situational_matches:
        primary = situational_matches[0]
        return applicable_sops, primary, True

    # Conflict rule: sort by severity descending
    sorted_sops = sorted(
        applicable_sops,
        key=lambda s: SEVERITY_WEIGHTS.get(s.severity.lower(), 0),
        reverse=True,
    )

    primary = sorted_sops[0]
    return sorted_sops, primary, False
