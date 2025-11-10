# reason2return (R2R) — Structured LLM Reasoning for Financial Decision Support

R2R turns heterogeneous financial data into structured analyst theses and disciplined trading recommendations (Strong Sell … Strong Buy).
The project implements a three-stage pipeline—Structure → Claims → Decision—trained with Supervised Fine-Tuning (SFT) and Reinforcement Fine-Tuning (RFT), including a pairwise ranking RL variant.
Scope includes data ingestion, training, backtesting, and an inference/API contract suitable for a research terminal.

This repository is for research and decision support, not high-frequency or fully autonomous trading.

**Positioning:** Analyst copilot for discretionary/quantamental workflows. Not high-frequency or fully autonomous order execution.

---

## Contents

- [Features](#features)
- [Success Metrics](#success-metrics)
- [Users & Use Cases](#users--use-cases)
- [Repository Layout](#repository-layout)
- [Quick Start](#quick-start)
- [Data & Labels](#data--labels)
- [Model Pipeline](#model-pipeline)
- [Reinforcement Learning](#reinforcement-learning)
- [Backtesting](#backtesting)
- [API (Contract)](#api-contract)
- [Configuration](#configuration)
- [System Architecture](#system-architecture)
- [Performance & SLAs](#performance--slas)
- [Evaluation Metrics](#evaluation-metrics)
- [Governance & Risk](#governance--risk)
- [Monitoring & Observability](#monitoring--observability)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License & Disclaimer](#license--disclaimer)

---

## Features

- Structured theses with required sections: market, fundamentals, sentiment, technicals, insider, risks.
- Evidence-grounded claims: each claim references a source snippet/quote and timestamp.
- Volatility-aware decisions: discrete labels with calibrated probabilities.
- RFT with composite rewards that balance structure, grounding, and market correctness.
- Pairwise ranking RL variant for stable credit assignment.
- Walk-forward backtesting with risk-adjusted metrics (Sharpe, max drawdown, etc.).
- Research terminal UI with thesis viewer, diff tracking, and scenario analysis.
- REST API for programmatic access to theses, decisions, and backtests.

---

## Success Metrics

The following targets must be met for production readiness:

### Performance Metrics

- **Sharpe ratio:** ≥ baseline +20% median improvement across test universe
- **Max drawdown:** ≤ baseline −15% median improvement
- **Hit rate:** Directional accuracy on strong buy/sell signals

### Quality Metrics

- **Claim coverage:** ≥ 90% of claims with at least one citation
- **Claim correctness:** ≥ 80% on auto-verifiable numeric/text match
- **Schema validity:** ≥ 95% of outputs pass JSON schema validation
- **Calibration (ECE):** ≤ 0.08 on decision probabilities

### Operational Metrics

- **Latency:** p95 ≤ 2.0s per ticker (with cached features)
- **Throughput:** 100 tickers/min/node (autoscale)
- **Uptime:** ≥ 99.5% for inference API
- **Reproducibility:** Artifact hashes, data snapshots, fixed seeds

---

## Users & Use Cases

### Personas

- **Buy-side analyst/PM:** Wants grounded theses and disciplined recommendations to support investment decisions
- **Quant researcher:** Needs reproducible pipeline, metrics, and APIs for systematic strategy development
- **Risk/compliance:** Requires traceability, audit logs, and data source attestations

### Primary Use Cases

- Daily or weekly thesis generation per coverage list
- Scenario analysis (macro shocks, earnings surprises)
- Signal review across names to drive portfolio rebalancing
- What-changed diffs between model versions or days
- Backtesting strategy variants with different parameters

---

## Repository Layout

Current structure:

```text
reason2return/
├─ notebooks/                                   # Reference Jupyter notebooks
│  ├─ Tiny_Trading_R1_RankingRL.ipynb          # Pairwise ranking RL demo
│  ├─ Tiny_Trading_R1_SchemaValidation.ipynb   # Schema-aware training
│  └─ README.md                                 # Notebook documentation
├─ r2r/                                        # Main package
│  ├─ data/                                    # Data loading & datasets
│  ├─ features/                                # Feature engineering
│  ├─ models/                                  # Model architectures
│  ├─ training/                                # Training loops (SFT, RFT)
│  ├─ backtest/                                # Backtesting engine
│  ├─ api/                                     # FastAPI service
│  └─ utils/                                   # Common utilities
├─ tests/                                      # Test suite
├─ configs/                                    # Configuration files
│  ├─ base_config.yaml                         # Base configuration
│  └─ experiments/                             # Experiment configs
├─ schemas/                                    # JSON schemas
│  ├─ thesis.schema.json                       # Thesis output format
│  └─ data_manifest.schema.json                # Data manifest format
├─ docs/                                       # Documentation
│  ├─ PRD.md                                   # Product requirements
│  ├─ MODULE_STRUCTURE.md                      # Code organization
│  ├─ PORTING_SUMMARY.md                       # Notebook→Production mapping
│  ├─ CODE_QUALITY_SETUP.md                    # Dev tools setup
│  └─ README.md                                # Documentation index
├─ pyproject.toml                              # Project metadata & config
├─ requirements.txt                            # Production dependencies
├─ requirements-dev.txt                        # Development dependencies
├─ .pre-commit-config.yaml                     # Pre-commit hooks
├─ CONTRIBUTING.md                             # Development guide
├─ TODO.md                                     # Current priorities
└─ README.md                                   # This file
```

**Key directories:**

- **`r2r/`** - Production Python package
- **`notebooks/`** - Reference implementations and experiments
- **`tests/`** - Test suite (pytest)
- **`configs/`** - YAML configuration files
- **`schemas/`** - JSON schemas for validation
- **`docs/`** - Detailed documentation
  │ └─ Tiny_Trading_R1_RankingRL.ipynb # Pairwise ranking RL toy demo
  ├─ r2r/
  │ ├─ data/ # ingestion & synthetic data generators
  │ ├─ features/ # feature builders & point-in-time joins
  │ ├─ models/ # backbones, heads, losses
  │ ├─ training/ # SFT/RFT trainers, loops
  │ ├─ backtest/ # walk-forward harness & metrics
  │ ├─ api/ # FastAPI service (optional)
  │ └─ utils/ # common tools, logging, config
  ├─ configs/
  │ ├─ training.yaml
  │ └─ backtest.yaml
  ├─ schemas/
  │ └─ thesis.schema.json
  ├─ tests/
  ├─ requirements.txt
  ├─ Makefile
  └─ README.md

````

The two toy notebooks under `notebooks/` provide end-to-end runnable demos.

---

## Quick Start

### 1) Environment

```bash
# Python 3.10 or 3.11 recommended
python -m venv .venv
source .venv/bin/activate
pip install -U pip

# minimal research stack
pip install -r requirements.txt
# If requirements.txt is not present yet, start with:
pip install torch numpy pandas scikit-learn matplotlib pydantic fastapi uvicorn mlflow rich pytest
````

### 2) Run the reference notebooks

Open the notebooks in [`notebooks/`](notebooks/):

- **`Tiny_Trading_R1_RankingRL.ipynb`** — Pairwise ranking RL implementation with SFT + RFT and walk-forward backtest.
- **`Tiny_Trading_R1_SchemaValidation.ipynb`** — Schema-aware training and validation approach.

Each notebook generates synthetic multi-ticker data, trains tiny models, and produces equity curves + performance metrics.

See [`notebooks/README.md`](notebooks/README.md) for details.

### 3) Run tests

```bash
# Run test suite
pytest

# Run with coverage
pytest --cov=r2r

# Tests are configured in pyproject.toml
# Coverage threshold: 20% (growing to 90%)
```

---

## Data & Labels

### Inputs (research mode)

- Prices (OHLCV)
- Fundamentals/filings (10-Q/10-K)
- News & sentiment
- Macro indicators
- Insider transactions
- Broker/consensus estimates

Start with synthetic data from notebooks. When moving to real data, enforce point-in-time joins and retain evidence snapshots.

**Storage:** Parquet data lake + feature store (time-series keyed by ticker, as_of) ensures strict event-time alignment and prevents look-ahead bias.

### Targets (decision labels)

For horizons $H = \{5, 21, 63\}$ trading days:

- $r_h = (P_{t+h} - P_t) / P_t$
- $v_t = \text{stdev}(\text{daily returns over lookback } W)$
- $z_h = r_h / (v_t+\epsilon)$, $\bar{z} = \text{mean}_h z_h$

Map $\bar{z}$ to 5 classes (configurable):

- Strong Sell ≤ −1.0
- Sell ≤ −0.25
- Hold (−0.25…0.25)
- Buy < 1.0
- Strong Buy ≥ 1.0

---

## Model Pipeline

### Stage I — Structure

Teach the model to output a strict schema:

- Sections: market, fundamentals, sentiment, technicals, insider, risks, and a table/quotes block.
- SFT-S: format and section boundaries.
- RFT-S reward: section presence, JSON validity, token budgets.

### Stage II — Claims

Every claim must be backed by an evidence snippet or numeric source with timestamp.

- SFT-C: <claim, evidence_span> pairs.
- RFT-C reward: coverage (claims with at least one citation), grounding (span/number match), hallucination penalty.

### Stage III — Decision

Map thesis to {SSell, Sell, Hold, Buy, SBuy}.

- SFT-D: supervised on volatility-normalized outcomes.
- RFT-D: composite reward (see below) with asymmetric penalties and optional smoothness term to reduce flip-flops.

### Optional distillation

- Thesis distillation from stronger models with reject sampling.
- Reverse reasoning distillation (decision → decomposed factors → compact thesis).

### Training Details

- **Backbone:** Reasoning-tuned 4–7B parameter model (e.g., Qwen-3 4B class)
- **Adaptation:** LoRA/QLoRA for parameter-efficient fine-tuning; BF16 where available
- **Curriculum:** Structure → Claims → Decision, followed by "mixture" clean-up pass
- **Data:** Point-in-time feature snapshots with strict event-time alignment

---

## Reinforcement Learning

### Composite reward (default)

$$R = w_S \cdot \text{Structure} + w_C \cdot \text{Claims} + w_D \cdot \text{Decision}$$

- Structure: fraction of required sections present + schema validity.
- Claims: coverage × correctness; penalize unsupported claims.
- Decision: asymmetric accuracy on class labels and volatility-normalized sign agreement.

Default weights (configurable): wS=0.2, wC=0.3, wD=0.5.

### GRPO-lite (batch-relative)

- Sample once per context; center rewards by batch mean to get an advantage baseline.

### Pairwise ranking RL (preferred for stability)

- For each context, sample two candidates.
- Rewards $R_1$, $R_2$ → zero-sum advantages $A_1=\tfrac{1}{2}(R_1-R_2)$, $A_2=-A_1$.
- Loss: maximize $A_1\log\pi(x_1) + A_2\log\pi(x_2)$ with entropy bonus and a small CE anchor on the decision head.

---

## Backtesting

### Walk-forward protocol

- Rolling train/test splits with strict time hygiene.
- Per-fold training: SFT warm-start → RFT.
- Trading rule (demo):
  - Strong Buy / Buy → long for 21 days
  - Strong Sell / Sell → short for 21 days
  - Hold → no position

### Transaction Costs & Constraints

- Per-name spreads (configurable)
- Slippage model based on order size
- Capacity constraints tied to average daily volume (ADV)
- Commission/fee modeling

### Metrics

- **Risk-adjusted:** Sharpe, Sortino, Calmar
- **Risk:** max drawdown, volatility, VaR
- **Behavior:** turnover, signal flip rate, holding period
- **Quality:** coverage/grounding of claims, calibration (ECE/Brier)
- **Capacity:** max position size vs ADV

### Statistical Testing

- Bootstrap confidence intervals for Sharpe ratio and max drawdown
- Diebold-Mariano test for forecast superiority
- Ablation studies: remove Claims reward, remove Structure, SFT-only, RL-only

Extend with costs (fees, spread, slippage), capacity (ADV), and portfolio constraints when moving beyond the demo.

---

## API (Contract)

The service returns structured theses and decisions. Example JSON:

```json
{
  "ticker": "MSFT",
  "as_of": "2025-01-15",
  "thesis": {
    "market": ["..."],
    "fundamentals": ["..."],
    "sentiment": ["..."],
    "technicals": ["..."],
    "insider": ["..."],
    "risks": ["..."],
    "evidence": [
      {
        "claim_id": "F1",
        "quote": "Revenue grew 12% YoY...",
        "source": "10-Q",
        "url": "https://example.com/filing",
        "time": "2024-10-30"
      }
    ]
  },
  "decision": {
    "label": "Buy",
    "probs": [0.06, 0.1, 0.22, 0.45, 0.17],
    "conviction": 0.67,
    "horizons_days": [21, 63]
  },
  "controls": {
    "max_position_pct_nav": 1.0,
    "stop_max_dd": 0.1
  },
  "explain": {
    "key_factors": ["EPS surprise", "margin expansion", "buyback"],
    "ablation_notes": "Removing sentiment reduces conviction by 0.15"
  }
}
```

### API Endpoints

A minimal FastAPI server can expose:

- **POST** `/v1/thesis` — single ticker inference (latest or as_of)
- **POST** `/v1/batch/thesis` — batch inference for multiple tickers
- **GET** `/v1/backtest` — run parameterized backtest (universe, horizon, costs)
- **GET** `/v1/healthz` — health check endpoint
- **GET** `/v1/metrics` — Prometheus-compatible metrics

---

## Configuration

Example `configs/training.yaml`:

```yaml
seed: 42
backbone: qwen-3-4b # placeholder; use your local backbone
optimizer:
  lr_sft: 3e-4
  lr_rft: 1e-4
rewards:
  wS: 0.2
  wC: 0.3
  wD: 0.5
  decision:
    asym_cost: { wrong_strong: 1.5, wrong_weak: 1.0 }
    smoothness_lambda: 0.1
rl:
  algorithm: pairwise_ranking # or grpo_lite
  entropy_coef: 0.001
  ce_anchor: 0.05
data:
  horizons: [5, 21, 63]
  vol_lookback: 60
  split:
    train_days: 320
    test_days: 120
```

Example `schemas/thesis.schema.json` (abbreviated):

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "R2R Thesis",
  "type": "object",
  "required": ["ticker", "as_of", "thesis", "decision"],
  "properties": {
    "ticker": { "type": "string" },
    "as_of": { "type": "string", "format": "date" },
    "thesis": {
      "type": "object",
      "required": [
        "market",
        "fundamentals",
        "sentiment",
        "technicals",
        "insider",
        "risks",
        "evidence"
      ],
      "properties": {
        "market": { "type": "array", "items": { "type": "string" } },
        "fundamentals": { "type": "array", "items": { "type": "string" } },
        "sentiment": { "type": "array", "items": { "type": "string" } },
        "technicals": { "type": "array", "items": { "type": "string" } },
        "insider": { "type": "array", "items": { "type": "string" } },
        "risks": { "type": "array", "items": { "type": "string" } },
        "evidence": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["claim_id", "quote", "source", "time"],
            "properties": {
              "claim_id": { "type": "string" },
              "quote": { "type": "string" },
              "source": { "type": "string" },
              "url": { "type": "string" },
              "time": { "type": "string", "format": "date" }
            }
          }
        }
      }
    },
    "decision": {
      "type": "object",
      "required": ["label", "probs"],
      "properties": {
        "label": {
          "type": "string",
          "enum": ["SSell", "Sell", "Hold", "Buy", "SBuy"]
        },
        "probs": {
          "type": "array",
          "items": { "type": "number" },
          "minItems": 5,
          "maxItems": 5
        },
        "conviction": { "type": "number" },
        "horizons_days": { "type": "array", "items": { "type": "integer" } }
      }
    }
  }
}
```

---

## System Architecture

### Pipeline Components

The system consists of several key components:

1. **Ingestion & Feature Store**

   - Parquet data lake (S3/GCS) for raw data
   - Feature store with point-in-time joins (keyed by ticker, as_of)
   - Vector store for evidence retrieval (optional)

2. **Training Pipeline**

   - Preprocessors: convert raw data → SFT corpora / RL buffers
   - Trainer: LoRA/PEFT for SFT; RL loop with ranking/GRPO on GPU
   - Model Registry: versioned artifacts, signatures, eval reports

3. **Inference Service**

   - Autoscaled, stateless serving layer
   - Feature cache for low-latency lookups
   - GPU optional (optimized for CPU inference)

4. **Observability**
   - Structured logs and distributed traces
   - Audit trail of evidence citations
   - Metrics export (Prometheus)

### Storage Architecture

- **Parquet lake:** Raw OHLCV, filings, news, sentiment feeds
- **Feature store:** Point-in-time aligned features per ticker
- **Vector store:** Evidence snippets for retrieval-augmented generation (optional)
- **Artifact store:** Versioned model checkpoints with signatures

### Security & Compliance

- VPC-isolated training environment
- KMS-encrypted storage buckets
- Signed model artifacts
- PII/MNPI classifiers on ingestion
- Data source allowlists and licensing enforcement

---

## Performance & SLAs

### Latency Targets

- **Single ticker inference:** p95 ≤ 2.0s (with cached features)
- **Batch processing:** 100 tickers/min/node (autoscale)
- **Feature cache hit rate:** ≥ 95% during market hours

### Reliability

- **Uptime:** ≥ 99.5% for inference API
- **Deployment strategy:** Blue/green deployments with automated rollback
- **Reproducibility:** All results deterministic given artifact hash + data snapshot + seed

### Scalability

- Horizontal autoscaling based on request volume
- Multi-region deployment (optional)
- Graceful degradation under load

---

## Evaluation Metrics

- Risk-adjusted: Sharpe, Sortino, Calmar
- Risk: max drawdown, volatility
- Behavioral: turnover, signal flip rate
- Quality: claim coverage, claim correctness (auto-verifiable), JSON validity
- Calibration: ECE, Brier score

### Baselines for Comparison

The system is evaluated against multiple baselines:

- **Buy-and-hold:** Passive equity holding
- **12-1 Momentum:** Classic momentum strategy
- **Analyst consensus:** Aggregated sell-side recommendations
- **SFT-only:** Model without reinforcement learning
- **Larger general LLM:** GPT-4 class model without fine-tuning

Target defaults (edit to your needs):

- Sharpe ≥ baseline +20% (median across names)
- Max drawdown ≤ baseline −15%
- Claim coverage ≥ 90%, claim correctness ≥ 80%
- ECE ≤ 0.08

---

## Governance & Risk

### Data Governance

- **Time hygiene:** Strict point-in-time joins; no look-ahead bias
- **Evidence snapshots:** Persist excerpts/ids with timestamps for audits
- **MNPI & licensing:** Use only licensed, public data; enforce data-source allowlists
- **Data lineage:** Track provenance from source → feature → model output

### Model Risk Management

- **Versioned artifacts:** All models signed with hash; evaluation reports attached
- **Drift monitoring:** Track data drift, feature drift, and prediction drift
- **Bias checks:** Regular audits for systematic biases across sectors/market caps
- **Rollback procedures:** Automated rollback triggers on metric degradation
- **A/B testing:** Shadow mode deployment before production cutover

### Compliance & Controls

- **Human-in-the-loop:** Analyst/PM reviews; model is decision support only, not autonomous trading
- **Audit logs:** Complete trail of model versions, data snapshots, and overrides
- **Disclaimers:** Clear messaging that output is for research/education, not investment advice
- **Access controls:** Role-based permissions for model deployment and data access
- **MNPI controls:** Automated classifiers to prevent material non-public information usage

### Risk Mitigation Strategies

| Risk                       | Mitigation                                                |
| -------------------------- | --------------------------------------------------------- |
| Regime shift / overfitting | Walk-forward validation, regular re-fit, robustness tests |
| Hallucinated evidence      | Schema validation, grounding rewards, retrieval snapshots |
| Data leakage               | Point-in-time joins, CI guards, independent audits        |
| Operational drift          | Monitoring dashboards, canary deployments, rollback       |
| Misuse as trading signal   | UI copy, disclaimers, approval gating                     |

---

## Roadmap

### P0 — Prototype (Weeks 0–4) ✅

**Status:** Complete

- Synthetic data + tiny nets
- Three-stage pipeline (Structure → Claims → Decision)
- GRPO-lite and pairwise ranking RL
- Basic backtester
- **Deliverables:** Toy notebooks, API mock, schema validator

### P1 — Research Alpha (Weeks 5–12)

**Goal:** Production-grade training and evaluation pipeline

- [ ] **TODO:** Integrate LLM (replace placeholder models with actual reasoning-tuned backbone)
- [ ] **TODO:** Connect real data sources (yfinance for prices, FRED for macro, news APIs for sentiment)
- Real data adapters (prices, filings, news/sentiment, insider)
- Retrieval-augmented evidence with citation highlighting
- SFT corpora generation
- Pairwise RL with full composite rewards
- [ ] **TODO:** Run backtests on historical data (full walk-forward with real market data)
- Full backtester with transaction costs and capacity limits
- Research terminal MVP (thesis viewer, diff view, backtest viewer)
- **Gate:** Metrics vs. baselines meet success targets; security review complete

### P2 — Pilot (Weeks 13–20)

**Goal:** Operational readiness

- Expanded universe coverage
- Calibration layer (temperature/Dirichlet scaling)
- Position-sizing head (continuous allocation)
- Risk controls and monitoring
- CI/CD pipeline
- Full observability stack (logs, traces, metrics)
- Human-in-the-loop workflows (PM approval gating)
- [ ] **TODO:** Deploy API to production (with autoscaling, monitoring, and SLAs)
- **Gate:** Reliability SLA met; compliance sign-off

### P3 — Limited GA (Weeks 21–28)

**Goal:** Production deployment

- Tenant isolation and access controls
- Model registry with promotion workflows
- Complete documentation and API references
- Customer onboarding materials
- Horizon selection optimization
- Terminal UI enhancements (scenario analysis, alerts)
- **Gate:** Production readiness review; customer validation

### Future Enhancements

- JSON-Schema validation + auto-repair for theses
- Cost/impact models with market impact estimation
- Multi-asset class support (bonds, commodities, FX)
- Options flow integration as evidence source
- Ensemble models with uncertainty decomposition

---

## Monitoring & Observability

### Offline Metrics

- Training losses per stage (Structure, Claims, Decision)
- Reward curves and component breakdowns
- Per-head metrics (schema validity, coverage, accuracy)
- Checkpoint evaluation reports

### Online Metrics

- **Latency:** p50, p95, p99 inference times
- **Error rates:** API errors, schema validation failures
- **Cache performance:** Hit rate, eviction rate
- **Quality drift:** % valid JSON, claim coverage, calibration shift
- **Resource usage:** GPU/CPU utilization, memory consumption

### Dashboards & Alerting

- **Prometheus/Grafana:** SLA tracking, system health
- **MLflow:** Experiment tracking, model comparison
- **Custom dashboards:** Thesis quality trends, decision distribution
- **Alerts:** Latency spikes, error rate increases, metric degradation

---

## Contributing

1. Open an issue describing the change or feature.
2. For code, include tests under `tests/` and ensure pytest passes.
3. Follow the JSON schema for outputs; add/adjust config under `configs/`.
4. Avoid adding any data that is not clearly licensed for public research use.

---

## License & Disclaimer

**License:** MIT (suggested—update if you prefer a different license).

**Disclaimer:** This software is for research and educational purposes only. It does not constitute investment advice. Past performance in backtests does not guarantee future results.

---

## Acknowledgements

This project implements a generalizable pattern—structured LLM reasoning + RL for market-aligned decisions—inspired by recent research on financial LLMs. The code here is original and simplified for clarity and reproducibility.

---

**Maintainer:** @Sinnema1

For questions or improvements, open an issue or pull request.
