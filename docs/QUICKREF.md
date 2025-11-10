# R2R Quick Reference

## Project Structure

```
reason2return/
├── r2r/                      # Main package
│   ├── data/                 # Data management
│   ├── features/             # Feature engineering
│   ├── models/               # Model architectures
│   ├── training/             # Training loops
│   ├── backtest/             # Backtesting
│   ├── api/                  # REST API
│   └── utils/                # Utilities
├── schemas/                  # JSON schemas
├── configs/                  # YAML configurations
├── tests/                    # Test suite
├── notebooks/                # Jupyter notebooks
└── MODULE_STRUCTURE.md       # Full documentation
```

## Common Tasks

### Generate Synthetic Data

```python
from r2r.data import SyntheticDataGenerator

gen = SyntheticDataGenerator(seed=42)
df = gen.generate_dataset(n_days=252, n_tickers=5, horizons=[5, 21, 63])
```

### Extract Features

```python
from r2r.features import FeatureExtractor, TechnicalIndicators

# Basic features
extractor = FeatureExtractor()
df = extractor.extract_all_features(df, price_cols=['TICK00', 'TICK01'])

# Technical indicators
tech = TechnicalIndicators()
df = tech.add_all_indicators(df, price_col='TICK00', prefix='TICK00')
```

### Validate Thesis

```python
from r2r.models import ThesisValidator

validator = ThesisValidator()
is_valid, errors = validator.validate(thesis_dict)
```

### Calculate Rewards

```python
from r2r.training import RewardCalculator

calc = RewardCalculator(w_structure=0.2, w_claims=0.3, w_decision=0.5)
rewards = calc.calculate_total_reward(thesis, actual_return=0.05)
```

### Run Backtest

```python
from r2r.backtest import BacktestEngine

engine = BacktestEngine(model, data, train_days=252, test_days=63)
results = engine.run_walk_forward()
metrics = engine.calculate_metrics()
```

### Start API Server

```bash
uvicorn r2r.api.app:create_app --reload --host 0.0.0.0 --port 8000
```

## Development Commands

```bash
# Activate environment
source .venv/bin/activate

# Run tests
pytest -v
pytest -v --cov=r2r --cov-report=html

# Code quality
ruff check r2r/ --fix
black r2r/
mypy r2r/

# Install package
pip install -e .
```

## Configuration Files

### Base Config

```yaml
# configs/base.yaml
model:
  hidden_size: 768
  num_layers: 12

sft:
  learning_rate: 5e-5
  batch_size: 16
  num_epochs: 3

rft:
  algorithm: "pairwise_ranking"
  learning_rate: 1e-5
  batch_size: 8
  num_epochs: 5
  w_structure: 0.2
  w_claims: 0.3
  w_decision: 0.5
```

### Experiment Config

```yaml
# configs/experiments/exp_001_baseline.yaml
experiment:
  name: "GRPO Baseline"
  description: "Standard GRPO with default parameters"

rft:
  algorithm: "grpo_lite"
```

## Key Classes

| Module         | Class                    | Purpose                                  |
| -------------- | ------------------------ | ---------------------------------------- |
| `r2r.data`     | `DataLoader`             | CSV I/O, manifest management             |
| `r2r.data`     | `Preprocessor`           | Scaling, missing values, lagged features |
| `r2r.data`     | `SyntheticDataGenerator` | Generate test data                       |
| `r2r.features` | `FeatureExtractor`       | Price, volume, momentum features         |
| `r2r.features` | `TechnicalIndicators`    | SMA, EMA, RSI, MACD, Bollinger           |
| `r2r.models`   | `ThesisPipeline`         | Three-stage model                        |
| `r2r.models`   | `ThesisValidator`        | Schema validation                        |
| `r2r.training` | `RewardCalculator`       | Three-component rewards                  |
| `r2r.training` | `SupervisedTrainer`      | SFT training loop                        |
| `r2r.training` | `ReinforceTrainer`       | RFT training loop                        |
| `r2r.backtest` | `BacktestEngine`         | Walk-forward validation                  |
| `r2r.backtest` | `PerformanceMetrics`     | Sharpe, drawdown, accuracy               |
| `r2r.api`      | `create_app()`           | FastAPI factory                          |

## Testing

### Run Specific Tests

```bash
pytest tests/test_schemas.py -v
pytest tests/test_config.py::test_base_config_exists -v
```

### Coverage Report

```bash
pytest --cov=r2r --cov-report=term-missing
```

### Test Fixtures

```python
# tests/conftest.py provides:
- synthetic_data()  # 100-day, 3-ticker dataset
- sample_thesis()   # Valid thesis dictionary
- device()          # CPU/CUDA device selection
```

## Logging

### Basic Usage

```python
from r2r.utils import get_logger, set_correlation_id

logger = get_logger(__name__)
correlation_id = set_correlation_id()  # For request tracing

logger.info("Processing started")
logger.warning("Missing data detected")
logger.error("Validation failed", extra={"thesis_id": "abc123"})
```

### Setup

```python
from r2r.utils import setup_logging

logger = setup_logging(
    level="INFO",
    log_format="structured",  # or "simple"
    log_file=Path("logs/app.log")
)
```

## API Endpoints

### POST /predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "date": "2025-11-09",
    "features": {"price": 150.0, "volume": 1000000}
  }'
```

### POST /validate

```bash
curl -X POST http://localhost:8000/validate \
  -H "Content-Type: application/json" \
  -d @thesis.json
```

### GET /health

```bash
curl http://localhost:8000/health
```

## Schema Format

### Thesis Structure

```json
{
  "ticker": "AAPL",
  "analysis_date": "2025-11-09",
  "thesis": {
    "market_environment": ["claim1", "claim2"],
    "fundamental_strength": ["claim1", "claim2"],
    "sentiment_analysis": ["claim1", "claim2"],
    "technical_setup": ["claim1", "claim2"],
    "insider_activity": ["claim1", "claim2"],
    "risk_factors": ["claim1", "claim2"]
  },
  "evidence": [
    {
      "claim_id": "market_1",
      "quote": "...",
      "source": "SEC Filing",
      "timestamp": "2025-11-08T10:00:00Z"
    }
  ],
  "decision": {
    "label": 3,
    "label_name": "buy",
    "probabilities": [0.05, 0.1, 0.2, 0.45, 0.2]
  }
}
```

## Next Steps Checklist

- [ ] Port notebook code to `ThesisPipeline`
- [ ] Integrate actual LLM (OpenAI, Anthropic, or local)
- [ ] Connect real data sources
- [ ] Run walk-forward backtest on historical data
- [ ] Deploy API to production
- [ ] Set up MLflow experiment tracking
- [ ] Add monitoring and alerting
