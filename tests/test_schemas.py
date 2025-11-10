"""Test schema validation utilities."""

import json

import pytest
from jsonschema import Draft202012Validator, ValidationError


def test_thesis_schema_exists(schema_dir):
    """Test that thesis schema file exists and is valid JSON."""
    schema_path = schema_dir / "thesis.schema.json"
    assert schema_path.exists(), "thesis.schema.json not found"

    with open(schema_path) as f:
        schema = json.load(f)

    assert "$schema" in schema
    assert "properties" in schema


def test_thesis_schema_validates_good_thesis(schema_dir, sample_thesis):
    """Test that a valid thesis passes schema validation."""
    schema_path = schema_dir / "thesis.schema.json"

    with open(schema_path) as f:
        schema = json.load(f)

    validator = Draft202012Validator(schema)

    # Should not raise
    validator.validate(sample_thesis)


def test_thesis_schema_rejects_missing_fields(schema_dir):
    """Test that thesis missing required fields fails validation."""
    schema_path = schema_dir / "thesis.schema.json"

    with open(schema_path) as f:
        schema = json.load(f)

    validator = Draft202012Validator(schema)

    # Missing 'decision' field
    incomplete_thesis = {
        "ticker": "AAPL",
        "as_of": "2025-11-09",
        "thesis": {
            "market": ["test"],
            "fundamentals": ["test"],
            "sentiment": ["test"],
            "technicals": ["test"],
            "insider": ["test"],
            "risks": ["test"],
            "evidence": [
                {
                    "claim_id": "C1",
                    "quote": "test",
                    "source": "test",
                    "time": "2025-11-09",
                }
            ],
        },
    }

    with pytest.raises(ValidationError):
        validator.validate(incomplete_thesis)


def test_thesis_schema_validates_decision_probs(schema_dir, sample_thesis):
    """Test that decision probs must be exactly 5 elements."""
    schema_path = schema_dir / "thesis.schema.json"

    with open(schema_path) as f:
        schema = json.load(f)

    validator = Draft202012Validator(schema)

    # Too few probs
    bad_thesis = sample_thesis.copy()
    bad_thesis["decision"]["probs"] = [0.5, 0.5]  # Only 2 instead of 5

    with pytest.raises(ValidationError):
        validator.validate(bad_thesis)


def test_data_manifest_schema_exists(schema_dir):
    """Test that data manifest schema exists."""
    schema_path = schema_dir / "data_manifest.schema.json"
    assert schema_path.exists(), "data_manifest.schema.json not found"

    with open(schema_path) as f:
        schema = json.load(f)

    assert "properties" in schema
    assert "data_sources" in schema["properties"]
