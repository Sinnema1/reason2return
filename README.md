reason2return (R2R) — Structured LLM Reasoning for Financial Decision Support

R2R turns heterogeneous financial data into structured analyst theses and disciplined trading recommendations (Strong Sell … Strong Buy).
The project implements a three-stage pipeline—Structure → Claims → Decision—trained with Supervised Fine-Tuning (SFT) and Reinforcement Fine-Tuning (RFT), including a pairwise ranking RL variant.
Scope includes data ingestion, training, backtesting, and an inference/API contract suitable for a research terminal.

This repository is for research and decision support, not high-frequency or fully autonomous trading.

⸻

Contents
	•	Features￼
	•	Repository Layout￼
	•	Quick Start￼
	•	Data & Labels￼
	•	Model Pipeline￼
	•	Reinforcement Learning￼
	•	Backtesting￼
	•	API (Contract)￼
	•	Configuration￼
	•	Metrics￼
	•	Governance & Risk￼
	•	Roadmap￼
	•	Contributing￼
	•	License & Disclaimer￼

⸻

Features
	•	Structured theses with required sections: market, fundamentals, sentiment, technicals, insider, risks.
	•	Evidence-grounded claims: each claim references a source snippet/quote and timestamp.
	•	Volatility-aware decisions: discrete labels with calibrated probabilities.
	•	RFT with composite rewards that balance structure, grounding, and market correctness.
	•	Pairwise ranking RL variant for stable credit assignment.
	•	Walk-forward backtesting with risk-adjusted metrics (Sharpe, max drawdown, etc.).

⸻

Repository Layout

Recommended structure (adopt incrementally):

reason2return/
├─ notebooks/
│  ├─ Tiny_Trading_R1_Pipeline.ipynb            # SFT + GRPO-lite toy demo
│  └─ Tiny_Trading_R1_RankingRL.ipynb           # Pairwise ranking RL toy demo
├─ r2r/
│  ├─ data/                                     # ingestion & synthetic data generators
│  ├─ features/                                 # feature builders & point-in-time joins
│  ├─ models/                                   # backbones, heads, losses
│  ├─ training/                                 # SFT/RFT trainers, loops
│  ├─ backtest/                                 # walk-forward harness & metrics
│  ├─ api/                                      # FastAPI service (optional)
│  └─ utils/                                    # common tools, logging, config
├─ configs/
│  ├─ training.yaml
│  └─ backtest.yaml
├─ schemas/
│  └─ thesis.schema.json
├─ tests/
├─ requirements.txt
├─ Makefile
└─ README.md

The two toy notebooks under notebooks/ provide end-to-end runnable demos.

⸻

Quick Start

1) Environment

# Python 3.10 or 3.11 recommended
python -m venv .venv
source .venv/bin/activate
pip install -U pip

# minimal research stack
pip install -r requirements.txt
# If requirements.txt is not present yet, start with:
pip install torch numpy pandas scikit-learn matplotlib pydantic fastapi uvicorn mlflow rich pytest

2) Run the toy demos

Open the notebooks in notebooks/:
	•	Tiny_Trading_R1_Pipeline.ipynb — three-stage pipeline with SFT + GRPO-lite RL and walk-forward backtest.
	•	Tiny_Trading_R1_RankingRL.ipynb — identical pipeline but pairwise ranking RL.

Each notebook generates a synthetic multi-ticker dataset, trains tiny heads, and produces an equity curve + Sharpe and max drawdown.

⸻

Data & Labels

Inputs (research mode)
	•	Prices (OHLCV)
	•	Fundamentals/filings (10-Q/10-K)
	•	News & sentiment
	•	Macro indicators
	•	Insider transactions

Start with synthetic data from notebooks. When moving to real data, enforce point-in-time joins and retain evidence snapshots.

Targets (decision labels)

For horizons H = \{5, 21, 63\} trading days:
	•	r_h = (P_{t+h} - P_t) / P_t
	•	v_t = \text{stdev}(\text{daily returns over lookback } W)
	•	z_h = r_h / (v_t+\epsilon), \bar{z} = \text{mean}_h z_h

Map \bar{z} to 5 classes (configurable):
	•	Strong Sell ≤ −1.0
	•	Sell ≤ −0.25
	•	Hold (−0.25…0.25)
	•	Buy < 1.0
	•	Strong Buy ≥ 1.0

⸻

Model Pipeline

Stage I — Structure

Teach the model to output a strict schema:
	•	Sections: market, fundamentals, sentiment, technicals, insider, risks, and a table/quotes block.
	•	SFT-S: format and section boundaries.
	•	RFT-S reward: section presence, JSON validity, token budgets.

Stage II — Claims

Every claim must be backed by an evidence snippet or numeric source with timestamp.
	•	SFT-C: <claim, evidence_span> pairs.
	•	RFT-C reward: coverage (claims with at least one citation), grounding (span/number match), hallucination penalty.

Stage III — Decision

Map thesis to {SSell, Sell, Hold, Buy, SBuy}.
	•	SFT-D: supervised on volatility-normalized outcomes.
	•	RFT-D: composite reward (see below) with asymmetric penalties and optional smoothness term to reduce flip-flops.

Optional distillation
	•	Thesis distillation from stronger models with reject sampling.
	•	Reverse reasoning distillation (decision → decomposed factors → compact thesis).

⸻

Reinforcement Learning

Composite reward (default)

R = w_S \cdot \text{Structure} + w_C \cdot \text{Claims} + w_D \cdot \text{Decision}
	•	Structure: fraction of required sections present + schema validity.
	•	Claims: coverage × correctness; penalize unsupported claims.
	•	Decision: asymmetric accuracy on class labels and volatility-normalized sign agreement.

Default weights (configurable): wS=0.2, wC=0.3, wD=0.5.

GRPO-lite (batch-relative)
	•	Sample once per context; center rewards by batch mean to get an advantage baseline.

Pairwise ranking RL (preferred for stability)
	•	For each context, sample two candidates.
	•	Rewards R_1, R_2 → zero-sum advantages A_1=\tfrac{1}{2}(R_1-R_2), A_2=-A_1.
	•	Loss: maximize A_1\log\pi(x_1) + A_2\log\pi(x_2) with entropy bonus and a small CE anchor on the decision head.

⸻

Backtesting

Walk-forward protocol
	•	Rolling train/test splits with strict time hygiene.
	•	Per-fold training: SFT warm-start → RFT.
	•	Trading rule (demo):
	•	Strong Buy / Buy → long for 21 days
	•	Strong Sell / Sell → short for 21 days
	•	Hold → no position

Metrics
	•	Risk-adjusted: Sharpe, Sortino, Calmar
	•	Risk: max drawdown, volatility
	•	Behavior: turnover, signal flip rate
	•	Quality: coverage/grounding of claims, calibration (ECE/Brier)

Extend with costs (fees, spread, slippage), capacity (ADV), and portfolio constraints when moving beyond the demo.

⸻

API (Contract)

The service returns structured theses and decisions. Example JSON:

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
    "probs": [0.06, 0.10, 0.22, 0.45, 0.17],
    "conviction": 0.67,
    "horizons_days": [21, 63]
  },
  "controls": { "max_position_pct_nav": 1.0, "stop_max_dd": 0.10 }
}

A minimal FastAPI server can expose:
	•	POST /v1/thesis — single ticker inference (latest or as_of).
	•	POST /v1/batch/thesis — batch inference.
	•	GET /v1/healthz, GET /v1/metrics — ops endpoints.

⸻

Configuration

Example configs/training.yaml:

seed: 42
backbone: qwen-3-4b   # placeholder; use your local backbone
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
  algorithm: pairwise_ranking   # or grpo_lite
  entropy_coef: 0.001
  ce_anchor: 0.05
data:
  horizons: [5, 21, 63]
  vol_lookback: 60
  split:
    train_days: 320
    test_days: 120

Example schemas/thesis.schema.json (abbreviated):

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
      "required": ["market","fundamentals","sentiment","technicals","insider","risks","evidence"],
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
        "label": { "type": "string", "enum": ["SSell", "Sell", "Hold", "Buy", "SBuy"] },
        "probs": { "type": "array", "items": { "type": "number" }, "minItems": 5, "maxItems": 5 },
        "conviction": { "type": "number" },
        "horizons_days": { "type": "array", "items": { "type": "integer" } }
      }
    }
  }
}


⸻

Metrics
	•	Risk-adjusted: Sharpe, Sortino, Calmar
	•	Risk: max drawdown, volatility
	•	Behavioral: turnover, signal flip rate
	•	Quality: claim coverage, claim correctness (auto-verifiable), JSON validity
	•	Calibration: ECE, Brier score

Target defaults (edit to your needs):
	•	Sharpe ≥ baseline +20% (median across names)
	•	Max drawdown ≤ baseline −15%
	•	Claim coverage ≥ 90%, claim correctness ≥ 80%
	•	ECE ≤ 0.08

⸻

Governance & Risk
	•	Time hygiene: point-in-time joins; no look-ahead.
	•	Evidence snapshots: persist excerpts/ids with timestamps for audits.
	•	MNPI & licensing: use only licensed, public data; enforce data-source allowlists.
	•	Model risk: versioned artifacts/evals, drift monitoring, rollback.
	•	Human-in-the-loop: analyst/PM reviews; model is decision support only.

⸻

Roadmap
	•	Real data adapters (prices, filings, news/sentiment, insider)
	•	Retrieval-augmented evidence with citation highlighting
	•	JSON-Schema validation + auto-repair for theses
	•	FastAPI inference service + batch jobs
	•	Cost/impact models in backtester; capacity limits (ADV)
	•	Position-sizing head (continuous) and horizon selection
	•	Calibration layer (temperature/Dirichlet)
	•	Terminal UI: thesis diffing, scenario analysis, and alerts

⸻

Contributing
	1.	Open an issue describing the change or feature.
	2.	For code, include tests under tests/ and ensure pytest passes.
	3.	Follow the JSON schema for outputs; add/adjust config under configs/.
	4.	Avoid adding any data that is not clearly licensed for public research use.

⸻

License & Disclaimer

License: MIT (suggested—update if you prefer a different license).
Disclaimer: This software is for research and educational purposes only. It does not constitute investment advice. Past performance in backtests does not guarantee future results.

⸻

Acknowledgements

This project implements a generalizable pattern—structured LLM reasoning + RL for market-aligned decisions—inspired by recent research on financial LLMs. The code here is original and simplified for clarity and reproducibility.

⸻

Maintainer: @Sinnema1
For questions or improvements, open an issue or pull request.
