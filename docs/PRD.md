This PRD defines the product boundary, measurable outcomes, and the technical/operational plan to reach a research-grade, auditable decision-support system for equity analysis and trading recommendations.

PRD: Trading-R1 — Structured LLM Reasoning for Financial Decision Support

1. Summary

Build and ship a decision-support system that turns heterogeneous financial data into structured analyst theses and disciplined trading recommendations (Strong Sell … Strong Buy). The product couples a three-stage reasoning pipeline (Structure → Claims → Decision) with supervised fine-tuning (SFT) and reinforcement fine-tuning (RFT)—including a pairwise ranking RL variant—to optimize for risk-adjusted returns while preserving interpretability. Scope includes data ingestion, model training, inference services, a research terminal UI, backtesting, and governance.

Positioning: Analyst copilot for discretionary/quantamental workflows. Not high-frequency or fully autonomous order execution.

⸻

2. Goals & Non-Goals

2.1 Goals
	•	Produce machine- and human-readable theses with strict structure and evidence citations.
	•	Convert theses into actionable, volatility-aware recommendations (5-class).
	•	Demonstrate superior risk-adjusted performance vs. baselines in walk-forward backtests.
	•	Provide a research terminal and API that standardize theses, claims, and decisions.
	•	Ensure governance: auditability, data lineage, MNPI controls, and model-risk management.

2.2 Success Metrics (must hit at P1)
	•	Sharpe ratio: ≥ baseline + 20% median improvement across test universe.
	•	Max drawdown: ≤ baseline − 15% median improvement.
	•	Claim coverage: ≥ 90% of claims with at least one citation.
	•	Claim correctness (auto-verifiable numeric/text match): ≥ 80%.
	•	Calibration (ECE on decision probabilities): ≤ 0.08.
	•	Latency (single ticker inference with cached features): ≤ 2.0s p95.
	•	Uptime: ≥ 99.5% for inference API.

2.3 Non-Goals
	•	Real-time order routing or HFT.
	•	Portfolio optimization & risk parity. (We expose hooks; not first-class scope.)
	•	Guaranteeing returns. This is decision support only.

⸻

3. Users & Use Cases

3.1 Personas
	•	Buy-side analyst/PM: wants grounded theses and disciplined recommendations.
	•	Quant researcher: wants reproducible pipeline, metrics, and APIs.
	•	Risk/compliance: needs traceability, audit logs, data source attestations.

3.2 Primary Use Cases
	•	Daily or weekly thesis generation per coverage list.
	•	Scenario analysis (macro shocks, earnings).
	•	Signal review across names to drive rebalancing.
	•	What-changed diffs between model versions or days.

⸻

4. Product Scope

4.1 Data Ingestion
	•	Sources: prices (OHLCV), fundamentals (filings), news, broker/consensus, macro, insider transactions, and sentiment feeds.
	•	Time alignment: strict event-time windows; deny future peeks; evidence snapshots retained.
	•	Storage: Parquet lake + feature store (time-series keyed by ticker, as_of).

4.2 Labeling & Targets
	•	Volatility-normalized outcomes
For horizons H = \{5d, 21d, 63d\}:
	•	r_h = \frac{P_{t+h}-P_t}{P_t}
	•	v_t = \text{stdev}(\text{daily returns over lookback } W)
	•	z_h = r_h / (v_t + \epsilon)
	•	\bar{z} = \text{mean}_h(z_h)
	•	Decision labels (5-class) by thresholds:
Strong Sell (≤ −1.0), Sell (≤ −0.25), Hold (−0.25…0.25), Buy (< 1.0), Strong Buy (≥ 1.0).
(Thresholds configurable per instrument class.)

4.3 Three-Stage Model Pipeline
	1.	Stage I — Structure
	•	Output must follow a strict schema: market, fundamentals, sentiment, technicals, insider, risks, conclusion, table/quotes.
	•	SFT-S on structured prompts to learn formatting and section boundaries.
	•	RFT-S reward: section presence, JSON validity, token budgets per section.
	2.	Stage II — Claims
	•	Every claim must link to evidence (quote or numeric source with timestamp).
	•	SFT-C on <claim, span> pairs and table extraction examples.
	•	RFT-C reward: coverage (claims with citations), grounding (semantic/number match), hallucination penalty.
	3.	Stage III — Decision
	•	Map thesis to {SSell, Sell, Hold, Buy, SBuy} with calibrated probabilities.
	•	SFT-D on labeled outcomes \bar{z}.
	•	RFT-D composite reward:
R = w_S \cdot \text{Structure} + w_C \cdot \text{Claims} + w_D \cdot \text{Decision}
	•	Decision component uses asymmetric cost matrix (punish wrong SSell/SBuy more).
	•	Add smoothness penalty (reduce flip-flops across adjacent days).

Distillation (optional but recommended):
	•	Thesis distillation from stronger models; reject-sampling filter.
	•	Reverse reasoning distillation: decompose decision → factors → compact thesis.

RL variants:
	•	GRPO-lite (group-relative baseline).
	•	Pairwise ranking RL: two samples per context; zero-sum advantages; entropy bonus; small CE anchor.

4.4 Inference Contract (JSON)

{
  "ticker": "MSFT",
  "as_of": "YYYY-MM-DD",
  "thesis": {
    "market": [...],
    "fundamentals": [...],
    "sentiment": [...],
    "technicals": [...],
    "insider": [...],
    "risks": [...],
    "evidence": [
      {"claim_id": "F1", "quote": "...", "source": "10-Q", "url": "…", "time": "YYYY-MM-DD"}
    ]
  },
  "decision": {"label": "Buy", "probs": [0.05,0.10,0.20,0.45,0.20], "conviction": 0.65, "horizons_days": [21,63]},
  "controls": {"max_position_pct_nav": 1.0, "stop_max_dd": 0.10},
  "explain": {"key_factors": ["EPS surprise", "margin expansion", "buyback"], "ablation_notes": "..."}
}

4.5 Research Terminal (MVP)
	•	Ticker view (thesis + evidence table + decision card).
	•	Diff view (day-over-day thesis/decision changes).
	•	What-if prompts (scenario rewrites; results flagged as “hypothetical”).
	•	Backtest viewer (equity curve, Sharpe, drawdown, turnover).

4.6 Backtesting & Simulation
	•	Walk-forward evaluation (rolling windows, multi-asset).
	•	Transaction costs: per-name spreads + slippage model; capacity constraints.
	•	Metrics: Sharpe, Sortino, Calmar, max DD, hit rate, turnover, capacity, stability (signal flip rate), calibration (ECE/Brier), coverage/grounding scores.

4.7 APIs
	•	POST /v1/thesis — returns structured thesis + decision for (ticker, as_of) or latest.
	•	POST /v1/batch/thesis — batch mode.
	•	GET /v1/backtest — run parameterized backtest (universe, horizon, costs).
	•	GET /v1/healthz, /v1/metrics (Prometheus).

⸻

5. System Architecture

Pipelines
	•	Ingestion & Feature Store → Preprocessors → SFT corpora / RL buffers.
	•	Trainer (LoRA/PEFT for SFT; RL loop with ranking/GRPO) on GPU.
	•	Model Registry (versioned artifacts, signatures, eval reports).
	•	Inference Service (autoscaled, stateless; feature cache; GPU optional).
	•	Observability (structured logs, traces, audit trail of evidence).

Storage
	•	Parquet lake (S3/GCS), Feature store (point-in-time joins), Vector store for evidence retrieval (optional).

Security
	•	VPC-isolated training; KMS-encrypted buckets; signed artifacts; PII/MNPI classifiers.

⸻

6. Detailed Requirements

6.1 Functional
	•	R1: Generate schema-valid theses with required sections (≥ 95% valid).
	•	R2: Each claim must include ≥1 evidence pointer (URL/id + timestamp).
	•	R3: Produce 5-class decision + probability vector; probabilities calibrated.
	•	R4: Backtester supports universe/period selection, costs, capacity.
	•	R5: Terminal UI shows thesis, evidence, decision, and diffs.
	•	R6: APIs return determinate JSON within SLA.

6.2 Non-Functional
	•	Latency: p95 ≤ 2.0s per ticker (w/ cached features).
	•	Throughput: 100 tickers/min/node (autoscale).
	•	Reliability: 99.5% uptime; blue/green deployments.
	•	Reproducibility: artifact hashes; data snapshots; fixed seeds.
	•	Privacy/Compliance: no MNPI; data licenses enforced; audit logs; disclaimers.

⸻

7. Training & Evaluation

7.1 SFT
	•	Backbone: reasoning-tuned 4–7B (Qwen-3 4B class).
	•	Adaptation: LoRA/QLoRA; BF16 where available.
	•	Curriculum: Structure → Claims → Decision; “mixture” clean-up pass.

7.2 RFT
	•	GRPO-lite: batch-relative centering; composite reward.
	•	Pairwise ranking RL (preferred for stability):
	•	Sample two candidates x_1, x_2 per context.
	•	Rewards R_1, R_2 via composite function.
	•	Advantages A_1=\tfrac{1}{2}(R_1-R_2), A_2=-A_1.
	•	Loss: -\mathbb{E}[A_1 \log\pi(x_1) + A_2 \log\pi(x_2)] - \alpha H(\pi) + \lambda \text{CE(anchor)}.

7.3 Backtesting Protocol
	•	Walk-forward: rolling train/test splits; strict time hygiene.
	•	Baselines: buy-and-hold, momentum, analyst consensus, smaller LLM, larger LLM.
	•	Stat tests: bootstrap CIs for Sharpe/DD; Diebold-Mariano for forecast loss.
	•	Ablations: remove Claims reward; remove Structure; SFT-only; RL-only.

⸻

8. Governance, Risk, and Compliance
	•	Disclaimers: “For research/education; not investment advice.”
	•	MNPI controls: data classifiers; allow-lists; ingestion contracts.
	•	Model risk: versioned evals, bias checks, drift monitoring, rollback.
	•	Human-in-the-loop: PM approval gating for live trials; audit log of overrides.
	•	Fair use: licensed data only; retain evidence snapshots for audits.

⸻

9. Telemetry & Monitoring
	•	Offline: training losses; reward curves; per-head metrics.
	•	Online: latency, error rates, cache hit rate, % valid JSON, claim coverage, calibration drift.
	•	Drift: data/feature drift detectors; alerting thresholds.
	•	Dashboards: Prometheus/Grafana (SLA), MLflow (experiments), Amplitude (UI).

⸻

10. Rollout Plan & Milestones

P0 — Prototype (Weeks 0–4)
	•	Synthetic data + tiny nets; three-stage pipeline; GRPO-lite; backtester.
	•	Deliverables: toy notebooks (done), API mock, schema validator.

P1 — Research Alpha (Weeks 5–12)
	•	Real data adapters; evidence retrieval; SFT corpora; pairwise RL; full backtester with costs; terminal MVP.
	•	Gate: metrics vs. baselines meet P1 targets; security review complete.

P2 — Pilot (Weeks 13–20)
	•	Expanded universe; calibration; risk controls; CI/CD; observability; human-in-the-loop workflows.
	•	Gate: reliability SLA; compliance sign-off.

P3 — Limited GA (Weeks 21–28)
	•	Tenant isolation; access controls; model registry promotions; documentation; customer onboarding.

⸻

11. Dependencies & Resourcing
	•	Data: price/quotes, fundamentals/filings, licensed news/sentiment, insider transactions.
	•	Infra: object store, feature store, GPU nodes for SFT/RL, model registry.
	•	People: 1 PM, 2–3 ML engineers, 1 data engineer, 1 backend, 1 frontend, 0.5 compliance.

⸻

12. Risks & Mitigations
	•	Regime shift / overfitting → walk-forward + regular re-fit + robustification tests.
	•	Hallucinated evidence → strict schema validation + claims grounding reward + retrieval snapshots.
	•	Data leakage → point-in-time joins; CI guards; audits.
	•	Operational drift → monitoring, canary deploys, rollback.
	•	Misuse as trading signal → UI copy, disclaimers, approval gating.

⸻

13. Acceptance Criteria (excerpt)

Area	Criteria
Structure	≥95% valid schema; each section present; JSON passes validator
Claims	≥90% coverage; ≥80% correctness on auto-verifiable items
Decision	ECE ≤ 0.08; confusion matrix not dominated by Hold; flip rate ≤ 25% WoW
Performance	Sharpe and max DD meet P1 targets across test names
API	p95 latency ≤ 2.0s; 99.5% uptime; idempotent responses
Governance	Audit log completeness; MNPI tests pass; reproducible runs


⸻

14. Open Questions
	•	Optimal decision thresholds per sector vs. global?
	•	Position sizing head (continuous) in P2?
	•	Should we learn horizon selection jointly with decision label?
	•	How to incorporate options flow as a claims/evidence source?
	•	Calibrated uncertainty decomposition (data vs. model vs. outcome noise)?

⸻

15. Appendices

A. Thesis JSON Schema (abbreviated)
	•	thesis.market[]: paragraphs with optional citations.
	•	thesis.evidence[]: {claim_id, quote, source_id/url, time, span_ref}.
	•	decision: {label ∈ {0..4}, probs[5], conviction ∈ [0,1]}.
	•	Strict JSON schema file to be included in repo (/schemas/thesis.schema.json).

B. Reward Weights (default)
	•	w_S = 0.2, w_C = 0.3, w_D = 0.5; entropy bonus \alpha = 1e\!-\!3; CE anchor \lambda = 0.05.
	•	Smoothness penalty \gamma = 0.1 on day-to-day decision changes.

C. Baselines
	•	Buy-and-hold; 12–1 momentum; analyst-consensus; SFT-only; larger general LLM.

D. Backtest Defaults
	•	Horizons: 5/21/63 days; costs: 2–5 bps + slippage model; capacity cap by ADV.

⸻

Deliverables already prepared
	•	Toy notebooks:
	•	Tiny three-stage GRPO-lite pipeline with walk-forward backtest.
	•	Pairwise ranking RL variant (two samples per context).
