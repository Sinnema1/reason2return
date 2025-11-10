# R2R Reference Notebooks

This directory contains reference Jupyter notebooks that demonstrate the R2R pipeline concepts.

## 📓 Available Notebooks

### [Tiny_Trading_R1_RankingRL.ipynb](Tiny_Trading_R1_RankingRL.ipynb)

**Purpose:** Demonstrates pairwise ranking RL for thesis generation

**Key Concepts:**

- `TinyPolicy` model architecture (encoder + 3 heads)
- `MarketToy` synthetic dataset
- Pairwise ranking RL training loop
- Walk-forward backtesting
- Performance evaluation

**What's Implemented:**

- Structure → Claims → Decision pipeline
- Teacher signal generation
- Supervised fine-tuning (SFT)
- Reinforcement fine-tuning (RFT) with pairwise advantage
- Equity curve and Sharpe ratio calculation

**Status:** ✅ Ported to production code (see `r2r/models/pipeline.py`, `r2r/training/rft.py`)

---

### [Tiny_Trading_R1_SchemaValidation.ipynb](Tiny_Trading_R1_SchemaValidation.ipynb)

**Purpose:** Demonstrates schema-aware training and validation

**Key Concepts:**

- JSON schema validation for thesis outputs
- Schema-aware reward functions
- Evidence grounding and citation tracking
- Structure completeness scoring

**What's Implemented:**

- `build_thesis_from_outputs()` - Converts model outputs to JSON
- `schema_valid()` - Validates against Draft202012Validator
- `structure_schema_score()` - Rewards valid schemas
- Claims truth evaluation

**Status:** ✅ Concepts integrated into reward calculator

---

## 🚀 Running the Notebooks

### Prerequisites

```bash
# Activate virtual environment
source .venv/bin/activate

# Install notebook dependencies
pip install jupyter ipykernel matplotlib

# Launch Jupyter
jupyter notebook
```

### What to Expect

- **Synthetic data generation** - No real market data required
- **Fast training** - Tiny models (< 1M parameters) for quick iteration
- **Complete pipeline** - SFT → RFT → Backtest in < 5 minutes
- **Visual outputs** - Equity curves, loss plots, performance metrics

---

## 📚 Relationship to Production Code

These notebooks served as the R&D sandbox. Key components have been ported:

| Notebook Component     | Production Module                      |
| ---------------------- | -------------------------------------- |
| `TinyPolicy`           | `r2r.models.pipeline.ThesisPipeline`   |
| `MarketToy`            | `r2r.data.dataset.ThesisDataset`       |
| `train_pairwise_rft()` | `r2r.training.rft.ReinforceTrainer`    |
| SFT training loop      | `r2r.training.sft.SupervisedTrainer`   |
| Schema validation      | `r2r.models.validator.ThesisValidator` |
| Walk-forward backtest  | `r2r.backtest.engine.BacktestEngine`   |

See [`docs/PORTING_SUMMARY.md`](../docs/PORTING_SUMMARY.md) for detailed mapping.

---

## 🔬 Experimental Notebooks

Create new experimental notebooks in `notebooks/experiments/` (gitignored):

```bash
mkdir -p notebooks/experiments
# Your experimental work goes here
```

Scratch work can go in `notebooks/scratch/` (also gitignored).

---

## 📝 Best Practices

When creating new notebooks:

1. **Add a clear title and purpose** at the top
2. **Document key parameters** and decisions
3. **Include example outputs** (plots, metrics)
4. **Note what worked and what didn't**
5. **Link to production code** if ported
6. **Keep cells executable** - don't leave broken cells
7. **Use markdown liberally** - explain your thinking

---

## 🎯 Next Steps

- [ ] Create `demo_training.ipynb` showing production pipeline usage
- [ ] Add notebook testing to CI (execute and validate outputs)
- [ ] Convert notebooks to mkdocs tutorials for documentation site

See [TODO.md](../TODO.md) for full roadmap.
