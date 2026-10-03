"""SOP Loader and Validator.

Discovers YAML SOP files, parses them, validates their schema and unique IDs,
and fails loudly at startup if any SOP is malformed.
"""

from pathlib import Path
from typing import Dict, List
import yaml
from pydantic import ValidationError

from backend.models import SOP

SUPPORTED_OPERATORS = {"gt", "gte", "lt", "lte", "eq", "in", "contains"}
SUPPORTED_SEVERITIES = {"low", "moderate", "high", "critical"}


class MalformedSOPError(Exception):
    """Raised when an SOP file is malformed, has invalid schema, or duplicate IDs."""
    pass


def _validate_conditions_block(conditions: dict, sop_id: str, file_path: Path) -> None:
    """Validate that condition operators are supported."""
    if not isinstance(conditions, dict):
        raise MalformedSOPError(
            f"File '{file_path.name}': SOP '{sop_id}' conditions must be a dictionary."
        )

    # Check 'all' and 'any' lists
    for key in ("all", "any"):
        if key in conditions:
            clause_list = conditions[key]
            if not isinstance(clause_list, list):
                raise MalformedSOPError(
                    f"File '{file_path.name}': SOP '{sop_id}' conditions['{key}'] must be a list."
                )
            for clause in clause_list:
                if not isinstance(clause, dict):
                    raise MalformedSOPError(
                        f"File '{file_path.name}': SOP '{sop_id}' condition item must be a dictionary."
                    )
                op = clause.get("operator")
                if op and op not in SUPPORTED_OPERATORS:
                    raise MalformedSOPError(
                        f"File '{file_path.name}': SOP '{sop_id}' uses unsupported operator '{op}'. "
                        f"Supported: {sorted(SUPPORTED_OPERATORS)}"
                    )


def load_sops_from_dir(sops_dir: Path) -> List[SOP]:
    """Load all SOPs from all .yaml and .yml files in the specified directory.

    Fails loudly with MalformedSOPError if any file fails validation or if
    duplicate SOP IDs are encountered.
    """
    if not sops_dir.exists() or not sops_dir.is_dir():
        raise MalformedSOPError(f"SOP directory does not exist: {sops_dir}")

    yaml_files = sorted(list(sops_dir.glob("*.yaml")) + list(sops_dir.glob("*.yml")))
    if not yaml_files:
        raise MalformedSOPError(f"No YAML SOP files found in directory: {sops_dir}")

    loaded_sops: List[SOP] = []
    seen_ids: Dict[str, Path] = {}

    for file_path in yaml_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f)
        except Exception as e:
            raise MalformedSOPError(f"Failed to parse YAML file '{file_path.name}': {e}") from e

        if not isinstance(content, list):
            raise MalformedSOPError(
                f"File '{file_path.name}' must contain a YAML list of SOP objects."
            )

        for item_idx, raw_item in enumerate(content):
            if not isinstance(raw_item, dict):
                raise MalformedSOPError(
                    f"File '{file_path.name}', item {item_idx} must be a dictionary."
                )

            # Pydantic validation
            try:
                sop = SOP(**raw_item)
            except ValidationError as ve:
                raise MalformedSOPError(
                    f"Validation failed for SOP in '{file_path.name}' item {item_idx}: {ve}"
                ) from ve

            # Severity check
            if sop.severity.lower() not in SUPPORTED_SEVERITIES:
                raise MalformedSOPError(
                    f"File '{file_path.name}': SOP '{sop.id}' has invalid severity '{sop.severity}'. "
                    f"Allowed: {sorted(SUPPORTED_SEVERITIES)}"
                )

            # Unique ID check
            if sop.id in seen_ids:
                raise MalformedSOPError(
                    f"Duplicate SOP ID '{sop.id}' detected in '{file_path.name}'. "
                    f"Already defined in '{seen_ids[sop.id].name}'."
                )
            seen_ids[sop.id] = file_path

            # Validate condition clauses if present
            if sop.conditions:
                _validate_conditions_block(sop.conditions, sop.id, file_path)

            loaded_sops.append(sop)

    return loaded_sops
