# Contributing to R2R (Reason2Return)

Thanks for your interest in contributing to R2R! This document provides guidelines for development.

## Development Setup

### Prerequisites

- Python 3.10 or 3.11
- macOS (primary development platform) or Linux

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/Sinnema1/reason2return.git
cd reason2return

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -U pip
pip install -e ".[dev]"

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

## Development Workflow

### 1. Code Quality Tools

We use several tools to maintain code quality:

- **Black**: Code formatting (100 char line length)
- **Ruff**: Linting and import sorting
- **mypy**: Type checking
- **pytest**: Testing with coverage

### 2. Pre-commit Hooks

Pre-commit hooks run automatically before each commit:

```bash
# Run manually on all files
pre-commit run --all-files

# Skip hooks (emergency only)
git commit --no-verify
```

Hooks include:

- Black formatting
- Ruff linting (with auto-fix)
- mypy type checking
- Bandit security checks
- Trailing whitespace, EOF fixes
- YAML/JSON validation

### 3. Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=r2r --cov-report=html

# Run specific test file
pytest tests/test_pipeline.py

# Run with verbose output
pytest -v

# Run only failed tests from last run
pytest --lf
```

**Coverage requirements:**

- P0: ≥70% line coverage
- P1: ≥80% line coverage
- P2: ≥90% line coverage

### 4. Code Style

#### Docstrings

We use **Google-style docstrings**:

```python
def train_model(data: pd.DataFrame, epochs: int = 10) -> dict:
    """Train the thesis generation model.

    Args:
        data: Training data with features and labels.
        epochs: Number of training epochs.

    Returns:
        Dictionary containing training metrics (loss, accuracy).

    Raises:
        ValueError: If data is empty or malformed.
    """
    pass
```

#### Type Hints

All public functions should have type hints:

```python
from typing import Dict, List, Optional, Tuple

def process_features(
    features: np.ndarray,
    labels: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Process features and return normalized values with stats."""
    pass
```

#### Import Order

Ruff automatically sorts imports in this order:

1. Standard library
2. Third-party packages
3. Local application imports

```python
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

from r2r.models.pipeline import ThesisPipeline
from r2r.utils.logging import get_logger
```

### 5. Testing Guidelines

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use fixtures for common setup
- Mock external dependencies (APIs, LLMs)
- Test edge cases and error conditions

Example test structure:

```python
import pytest
import torch

from r2r.models.pipeline import ThesisPipeline


@pytest.fixture
def model():
    """Create a test model."""
    config = {"input_dim": 10, "hidden_dim": 32}
    return ThesisPipeline(config=config)


def test_forward_pass(model):
    """Test model forward pass returns correct shapes."""
    x = torch.randn(4, 10)
    outputs = model(x)

    assert "structure_logits" in outputs
    assert outputs["structure_logits"].shape == (4, 6)
```

## Making Changes

### Branch Naming

- Feature: `feature/add-llm-integration`
- Bugfix: `fix/backtest-crash`
- Docs: `docs/update-contributing`

### Commit Messages

Use conventional commits format:

```
feat: add pairwise ranking RL trainer
fix: correct tensor shapes in claims head
docs: update API documentation
test: add integration tests for pipeline
refactor: simplify reward calculation
chore: update dependencies
```

### Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with tests
3. Ensure all tests pass: `pytest`
4. Ensure pre-commit hooks pass: `pre-commit run --all-files`
5. Update documentation if needed
6. Submit PR with clear description

## Project Structure

```
reason2return/
├── r2r/                    # Main package
│   ├── data/              # Data loading and datasets
│   ├── features/          # Feature engineering
│   ├── models/            # Model architectures
│   ├── training/          # Training loops (SFT, RFT)
│   ├── backtest/          # Backtesting engine
│   ├── api/               # FastAPI service
│   └── utils/             # Common utilities
├── tests/                 # Test suite
├── notebooks/             # Jupyter notebooks
├── configs/               # Configuration files
├── schemas/               # JSON schemas
└── docs/                  # Documentation (future)
```

## Code Review Checklist

Before submitting, ensure:

- [ ] Code passes all pre-commit hooks
- [ ] All tests pass (`pytest`)
- [ ] Coverage doesn't decrease (≥70% currently)
- [ ] Type hints on public functions
- [ ] Docstrings on public functions/classes
- [ ] Updated README/docs if needed
- [ ] No TODOs in production code (move to TODO.md)
- [ ] No sensitive data (API keys, credentials)

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for general questions
- Check existing issues before creating new ones

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
