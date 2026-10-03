"""Unit tests for SOP loader and validation logic."""

from pathlib import Path
import pytest
from backend.loader import load_sops_from_dir, MalformedSOPError

SOPS_DIR = Path(__file__).resolve().parent.parent / "sops"


def test_load_valid_sops():
    """Verify that legitimate SOP files load and meet assignment requirements."""
    sops = load_sops_from_dir(SOPS_DIR)
    assert len(sops) >= 10, f"Expected at least 10 SOPs, found {len(sops)}"

    categories = {sop.category for sop in sops}
    assert len(categories) >= 3, f"Expected at least 3 categories, found {len(categories)}"

    # Verify presence of situational SOP
    situational_sops = [s for s in sops if s.situational]
    assert len(situational_sops) >= 1, "Must contain at least 1 situational override SOP"

    # Verify IDs are unique
    ids = [s.id for s in sops]
    assert len(ids) == len(set(ids)), "SOP IDs must be strictly unique"


def test_reject_duplicate_ids(tmp_path):
    """Verify that duplicate SOP IDs fail loudly."""
    dup_yaml = tmp_path / "dup.yaml"
    dup_yaml.write_text(
        """
- id: SOP-TEST-01
  category: test
  severity: low
  description: Rule 1
  advice: Test advice 1
  cite_as: SOP-TEST-01
- id: SOP-TEST-01
  category: test
  severity: high
  description: Rule 2
  advice: Test advice 2
  cite_as: SOP-TEST-01
        """,
        encoding="utf-8",
    )
    with pytest.raises(MalformedSOPError) as exc_info:
        load_sops_from_dir(tmp_path)
    assert "Duplicate SOP ID" in str(exc_info.value)


def test_reject_malformed_yaml(tmp_path):
    """Verify that syntax errors in YAML fail loudly with file name."""
    bad_yaml = tmp_path / "bad_syntax.yaml"
    bad_yaml.write_text("invalid: [yaml: syntax error: }", encoding="utf-8")
    with pytest.raises(MalformedSOPError) as exc_info:
        load_sops_from_dir(tmp_path)
    assert "Failed to parse YAML file" in str(exc_info.value)


def test_reject_invalid_severity(tmp_path):
    """Verify that unpermitted severity values fail loudly."""
    bad_severity = tmp_path / "bad_sev.yaml"
    bad_severity.write_text(
        """
- id: SOP-TEST-02
  category: test
  severity: super_dangerous
  description: Invalid severity test
  advice: Do not proceed
  cite_as: SOP-TEST-02
        """,
        encoding="utf-8",
    )
    with pytest.raises(MalformedSOPError) as exc_info:
        load_sops_from_dir(tmp_path)
    assert "invalid severity" in str(exc_info.value)


def test_reject_unsupported_operator(tmp_path):
    """Verify that unsupported condition operators fail loudly."""
    bad_op = tmp_path / "bad_op.yaml"
    bad_op.write_text(
        """
- id: SOP-TEST-03
  category: test
  severity: high
  description: Bad operator test
  conditions:
    all:
      - field: wind_speed_10m
        operator: bitwise_xor
        value: 12
  advice: Do not proceed
  cite_as: SOP-TEST-03
        """,
        encoding="utf-8",
    )
    with pytest.raises(MalformedSOPError) as exc_info:
        load_sops_from_dir(tmp_path)
    assert "unsupported operator" in str(exc_info.value)
