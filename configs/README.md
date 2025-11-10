# Configuration Management

This directory contains YAML configuration files for training, backtesting, and experiments.

## Structure

```
configs/
├── base.yaml                    # Default configuration (all settings)
├── experiments/                 # Experiment-specific overrides
│   ├── exp_001_grpo_baseline.yaml
│   ├── exp_002_pairwise_ranking.yaml
│   └── exp_003_schema_validation.yaml
└── README.md
```

## Usage

### Loading Configurations

```python
import yaml
from pathlib import Path

def load_config(experiment_name=None):
    """Load base config, optionally merged with experiment config."""
    config_dir = Path("configs")

    # Load base
    with open(config_dir / "base.yaml") as f:
        config = yaml.safe_load(f)

    # Merge experiment overrides
    if experiment_name:
        exp_path = config_dir / "experiments" / f"{experiment_name}.yaml"
        with open(exp_path) as f:
            exp_config = yaml.safe_load(f)
        config = deep_merge(config, exp_config)

    return config

# Example
config = load_config("exp_002_pairwise_ranking")
```

### Environment Variables

Sensitive settings (API keys, paths) should use environment variables:

```yaml
# In config file
paths:
  data: ${DATA_DIR:-data} # Default to 'data' if not set
```

```python
import os
from string import Template

def expand_env_vars(config_dict):
    """Recursively expand environment variables in config."""
    # Implementation left as exercise
    pass
```

## Configuration Hierarchy

1. **base.yaml** - All default settings
2. **experiment/\*.yaml** - Overrides for specific experiments
3. **Environment variables** - Runtime overrides (paths, secrets)
4. **CLI arguments** (future) - One-off parameter changes

## Adding New Experiments

1. Copy a similar experiment file
2. Update `experiment.name` and `experiment.description`
3. Override only the settings you want to change
4. Add descriptive tags for organization

## Best Practices

### ✅ Do:

- Keep `base.yaml` comprehensive with sensible defaults
- Document unusual parameter choices in experiment configs
- Use semantic versioning for breaking config changes
- Tag experiments for easy filtering

### ❌ Don't:

- Don't put secrets in config files (use env vars)
- Don't duplicate settings across experiment files
- Don't use absolute paths (keep configs portable)
- Don't commit local developer overrides

## Config Validation

Future: Add Pydantic models to validate configs at runtime:

```python
from pydantic import BaseModel, validator

class ModelConfig(BaseModel):
    backbone: str
    hidden_dim: int
    n_sections: int = 6

    @validator('hidden_dim')
    def hidden_dim_positive(cls, v):
        if v <= 0:
            raise ValueError('hidden_dim must be positive')
        return v
```

## Version Control

- ✅ Commit: All config files in `configs/`
- ❌ Don't commit: Local overrides, secrets, absolute paths
- 📝 Document: Breaking changes in experiment descriptions
