# Schema Definitions

This directory contains JSON schemas that define data contracts for the R2R system.

## Schemas

### `thesis.schema.json`

Defines the structure for model outputs (theses with evidence + decisions).

**Key sections:**

- `thesis`: 6 required sections (market, fundamentals, sentiment, technicals, insider, risks) + evidence array
- `decision`: 5-class label with probability distribution
- `controls`: Optional risk parameters
- `explain`: Optional explainability metadata

**Validation:**

```python
from jsonschema import Draft202012Validator
import json

with open('schemas/thesis.schema.json') as f:
    schema = json.load(f)

validator = Draft202012Validator(schema)
validator.validate(thesis_obj)  # Raises exception if invalid
```

### `data_manifest.schema.json`

Defines metadata for dataset snapshots to ensure reproducibility.

**Key sections:**

- `data_sources`: Describes each data source (type, date range, hash)
- `features`: Feature engineering parameters
- `labels`: Label generation methodology
- `hash`: SHA-256 of entire dataset for versioning

**Usage:**
Create a manifest file when generating training data:

```python
manifest = {
    "manifest_version": "1.0",
    "created_at": "2025-11-09T12:00:00Z",
    "data_sources": [...],
    "hash": "abc123..."
}
```

## Version Policy

- Schema files are versioned (e.g., `thesis.v1.schema.json`)
- Breaking changes require new version
- Backward-compatible changes can update existing version
- API endpoints should specify schema version in responses

## Future Schemas

Planned additions:

- `feature_store.schema.json` - Point-in-time feature schema
- `backtest_config.schema.json` - Backtest parameter validation
- `model_card.schema.json` - Model artifact metadata
