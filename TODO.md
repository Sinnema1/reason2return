# R2R Project TODOs

## Current Status

✅ **Prototype Complete** — Three-stage pipeline ported from notebooks to production code with full test coverage (16 passed, 1 skipped)

---

## Priority 1: Research Alpha (Weeks 5–12)

### 🔴 Integrate LLM (Replace Placeholder Models)

**Current State:** Using simple `ThesisPipeline` with Linear layers as proof-of-concept

**Goal:** Integrate actual reasoning-tuned LLM backbone

**Tasks:**

- [ ] Select backbone model (Qwen-3 4B, Llama-3 8B, or similar)
- [ ] Implement tokenization and text generation pipeline
- [ ] Add LoRA/QLoRA adapters for parameter-efficient fine-tuning
- [ ] Update `ThesisPipeline` to use transformer backbone
- [ ] Implement retrieval-augmented generation (RAG) for evidence
- [ ] Add citation highlighting and evidence tracking
- [ ] Test generation quality on sample theses

**Dependencies:** None

**Estimated Effort:** 2-3 weeks

---

### 🔴 Connect Real Data Sources

**Current State:** Using synthetic data from `SyntheticDataGenerator`

**Goal:** Integrate production data sources with point-in-time integrity

**Tasks:**

#### Price Data (yfinance)

- [ ] Create `YFinanceAdapter` in `r2r/data/loader.py`
- [ ] Implement OHLCV data fetching with error handling
- [ ] Add data validation and gap filling
- [ ] Store in Parquet format with proper indexing
- [ ] Test with multiple tickers and date ranges

#### Macroeconomic Data (FRED)

- [ ] Create `FREDAdapter` for macro indicators
- [ ] Fetch key series (GDP, CPI, unemployment, rates)
- [ ] Implement point-in-time alignment with price data
- [ ] Handle data release delays and revisions
- [ ] Cache for performance

#### News & Sentiment

- [ ] Integrate news API (NewsAPI, Alpha Vantage, or Benzinga)
- [ ] Implement sentiment scoring (transformer-based or commercial)
- [ ] Associate news with tickers and timestamps
- [ ] Create evidence snippets with source attribution
- [ ] Handle rate limits and pagination

#### SEC Filings

- [ ] Create SEC EDGAR scraper for 10-Q/10-K filings
- [ ] Parse filing text and extract key sections
- [ ] Implement filing date tracking (filed vs period-end)
- [ ] Create searchable evidence store
- [ ] Add citation extraction

#### Insider Transactions

- [ ] Scrape insider transaction data (SEC Form 4)
- [ ] Parse transaction types (buy/sell) and amounts
- [ ] Link to tickers and officers
- [ ] Calculate insider sentiment signals

#### Data Pipeline

- [ ] Implement `DataOrchestrator` to coordinate all sources
- [ ] Create point-in-time feature store with strict event-time joins
- [ ] Add data quality checks and validation
- [ ] Implement caching and incremental updates
- [ ] Write integration tests for data pipeline

**Dependencies:** None

**Estimated Effort:** 3-4 weeks

---

### 🔴 Run Backtests on Historical Data

**Current State:** Basic backtester exists but uses synthetic data

**Goal:** Full walk-forward validation with real market data

**Tasks:**

- [ ] Update `BacktestEngine` to work with real data sources
- [ ] Implement transaction cost model:
  - [ ] Bid-ask spreads (configurable by ticker/liquidity)
  - [ ] Slippage model based on order size vs ADV
  - [ ] Commission/fee structure
- [ ] Add capacity constraints:
  - [ ] Max position size as % of ADV
  - [ ] Portfolio concentration limits
  - [ ] Sector exposure limits
- [ ] Implement walk-forward protocol:
  - [ ] Rolling train/test splits with time hygiene
  - [ ] SFT → RFT training per fold
  - [ ] Out-of-sample performance tracking
- [ ] Add advanced metrics:
  - [ ] Sharpe, Sortino, Calmar ratios
  - [ ] Maximum drawdown and duration
  - [ ] Win rate and profit factor
  - [ ] Turnover and holding period analysis
- [ ] Create backtest comparison framework:
  - [ ] Compare vs baselines (buy-and-hold, momentum, consensus)
  - [ ] Statistical significance testing (bootstrap, Diebold-Mariano)
  - [ ] Ablation studies (SFT-only, no Claims, etc.)
- [ ] Generate backtest reports:
  - [ ] Equity curves and drawdown charts
  - [ ] Performance tables and tear sheets
  - [ ] Trade-by-trade logs
  - [ ] Risk attribution analysis

**Dependencies:** Real data sources connected

**Estimated Effort:** 2-3 weeks

---

## Priority 2: Pilot (Weeks 13–20)

### 🟡 Deploy API to Production

**Current State:** Basic FastAPI skeleton in `r2r/api/app.py`

**Goal:** Production-ready API service with monitoring and SLAs

**Tasks:**

#### API Implementation

- [ ] Complete FastAPI endpoints:
  - [ ] `POST /v1/thesis` — Single ticker inference
  - [ ] `POST /v1/batch/thesis` — Batch inference
  - [ ] `GET /v1/backtest` — Run parameterized backtest
  - [ ] `GET /v1/healthz` — Health check
  - [ ] `GET /v1/metrics` — Prometheus metrics
- [ ] Add request validation and error handling
- [ ] Implement authentication/authorization (API keys)
- [ ] Add rate limiting and throttling
- [ ] Write API documentation (OpenAPI/Swagger)

#### Inference Optimization

- [ ] Implement feature caching layer (Redis/Memcached)
- [ ] Optimize model loading and batching
- [ ] Add GPU acceleration (optional)
- [ ] Profile and optimize latency (target p95 ≤ 2.0s)
- [ ] Implement request queuing for load management

#### Infrastructure

- [ ] Containerize with Docker
- [ ] Create Kubernetes deployment manifests
- [ ] Set up horizontal autoscaling (HPA)
- [ ] Configure load balancer and ingress
- [ ] Implement blue/green deployment strategy
- [ ] Add automated rollback on health check failures

#### Monitoring & Observability

- [ ] Integrate Prometheus for metrics collection
- [ ] Create Grafana dashboards:
  - [ ] Latency (p50, p95, p99)
  - [ ] Error rates and types
  - [ ] Cache hit rates
  - [ ] Request volume and throughput
  - [ ] Resource utilization (CPU, memory, GPU)
- [ ] Set up alerting (PagerDuty/Opsgenie):
  - [ ] Latency SLA violations (p95 > 2.0s)
  - [ ] Error rate spikes (> 1%)
  - [ ] Service downtime
- [ ] Implement structured logging (JSON logs)
- [ ] Add distributed tracing (Jaeger/Zipkin)
- [ ] Create audit trail for compliance

#### Testing & Validation

- [ ] Write comprehensive API tests
- [ ] Implement load testing (k6/Locust)
- [ ] Validate SLA compliance:
  - [ ] p95 latency ≤ 2.0s
  - [ ] Uptime ≥ 99.5%
  - [ ] Throughput ≥ 100 tickers/min/node
- [ ] Conduct security audit
- [ ] Perform disaster recovery drills

**Dependencies:** Real data integration, backtest validation complete

**Estimated Effort:** 3-4 weeks

---

## Priority 3: Near-Term Improvements

### 📝 Create Example Training Notebook

- [ ] Create `notebooks/demo_training.ipynb`
- [ ] Demonstrate end-to-end workflow:
  - [ ] Load synthetic data
  - [ ] Create ThesisDataset with teacher signals
  - [ ] Train with SFT for 2 epochs
  - [ ] Fine-tune with RFT for 2 epochs
  - [ ] Run walk-forward backtest
  - [ ] Generate performance plots
- [ ] Add explanatory markdown cells
- [ ] Include hyperparameter tuning examples

**Estimated Effort:** 3-5 days

---

### � Improve Test Coverage

**Current:** 22% line coverage
**Target:** P0: 40% → P1: 70% → P2: 90%

- [ ] Add unit tests for data loaders (currently 20-38%)
- [ ] Add unit tests for feature extractors (currently 0%)
- [ ] Add unit tests for reward calculator (currently 13%)
- [ ] Add unit tests for trainers (currently 16%)
- [ ] Add integration tests for backtest engine
- [ ] Add property-based tests with hypothesis (optional)

**Estimated Effort:** Ongoing, 2-3 days for P0 target

---

### �🔧 Update Backtest Engine for ThesisPipeline

- [ ] Modify `BacktestEngine.run_single_window()` to use `model.predict()`
- [ ] Handle batch predictions efficiently
- [ ] Extract decision labels and probabilities
- [ ] Update position sizing based on conviction
- [ ] Test with trained ThesisPipeline models

**Estimated Effort:** 2-3 days

---

### 📖 Document Reward Calculator Integration

- [ ] Create `docs/REWARDS.md` explaining reward components
- [ ] Document how rewards connect to RFT training
- [ ] Provide examples of custom reward functions
- [ ] Add configuration guide for reward weights
- [ ] Include ablation study results

**Estimated Effort:** 2-3 days

---

## Priority 4: Future Enhancements

### Code Quality & DevOps (P1)

#### CI/CD Pipeline

- [ ] Set up GitHub Actions workflow
  - [ ] Matrix testing: Python 3.10, 3.11 on macOS
  - [ ] Run pytest, ruff, black --check, mypy
  - [ ] Generate coverage reports
  - [ ] Build documentation (when ready)
- [ ] Add branch protection rules
- [ ] Configure automated dependency updates (Dependabot)

**Estimated Effort:** 1-2 days

#### Documentation Site (P1/P2)

- [ ] Set up MkDocs with Material theme
- [ ] Configure mkdocstrings for API docs
- [ ] Convert notebooks to tutorials
- [ ] Auto-generate API reference from docstrings
- [ ] Add version selector
- [ ] Deploy to GitHub Pages

**Estimated Effort:** 3-5 days

#### Stricter Type Checking (P1)

- [ ] **Fix PyTorch type issues in training modules** (CRITICAL - currently blocking mypy)
  - [ ] Add proper type stubs for `ThesisPipeline.sample()` method
  - [ ] Fix Tensor vs Module union type issues in `RFTTrainer._build_thesis_from_samples()`
  - [ ] Add `calculate_batch_rewards()` method to `RewardCalculator` or fix type hints
  - [ ] Re-enable mypy pre-commit hook after fixes
- [ ] Enable `warn_return_any = true` in mypy (currently disabled)
- [ ] Enable `disallow_untyped_defs = true` in mypy
- [ ] Enable `disallow_any_generics = true` in mypy
- [ ] Add type stubs for third-party packages
- [ ] Achieve 100% type coverage on public API
- [ ] Enable stricter Ruff rules (currently ignoring UP007, N806, C901, NPY002)

**Estimated Effort:** Ongoing, 2-3 days initial push

**Note:** Mypy is currently disabled in pre-commit due to complex PyTorch typing issues. See `.pre-commit-config.yaml` for details.

#### Security & Compliance (P2)

- [ ] Add `pip-audit` to CI for vulnerability scanning
- [ ] Configure Dependabot security alerts
- [ ] Add SECURITY.md with disclosure policy
- [ ] Implement secrets scanning (GitHub Advanced Security)

**Estimated Effort:** 1 day

---

### Advanced Features

- [ ] Schema validation in `generate_thesis()` with auto-repair
- [ ] Configurable claim heuristics (YAML-driven)
- [ ] Multi-task loss weighting (learnable weights)
- [ ] Curriculum learning for RFT (progressive difficulty)
- [ ] Ensemble predictions (multiple model checkpoints)
- [ ] Temperature scaling for calibration
- [ ] Uncertainty quantification (epistemic + aleatoric)

### Data & Features

- [ ] Options flow as evidence source
- [ ] Social media sentiment (Twitter/Reddit)
- [ ] Earnings call transcripts
- [ ] Analyst upgrades/downgrades
- [ ] Short interest data
- [ ] ETF flows and positioning

### Model Improvements

- [ ] Multi-horizon predictions (5/21/63 days)
- [ ] Position sizing head (continuous allocation)
- [ ] Sector-aware embeddings
- [ ] Cross-asset relationships
- [ ] Regime detection module

### Tooling

- [ ] Research terminal UI (Streamlit/Gradio)
  - [ ] Thesis viewer with diff tracking
  - [ ] Interactive backtests
  - [ ] Scenario analysis
  - [ ] Alert configuration
- [ ] Model registry (MLflow/Weights & Biases)
- [ ] Experiment tracking dashboard
- [ ] A/B testing framework
- [ ] Synthetic data generator improvements

---

## Completed ✅

### Code Quality Setup (Just Completed!)

- [x] Pre-commit hooks configuration (.pre-commit-config.yaml)
- [x] Enhanced Ruff config with docstring checks (D rules)
- [x] Stricter mypy configuration (warn_redundant_casts, warn_unused_ignores)
- [x] Public API definition in r2r/**init**.py (**all**)
- [x] CONTRIBUTING.md with development workflow
- [x] Coverage threshold in pytest (70%, will raise to 80%/90%)

### Initial Setup

- [x] Module structure (8 modules, 24 files)
- [x] Schema definitions (thesis, data manifest)
- [x] Configuration system (base + experiments)
- [x] Dependency management (requirements.txt, requirements-dev.txt)
- [x] Logging infrastructure
- [x] Test framework setup (pytest)
- [x] ThesisPipeline model (encoder + 3 heads)
- [x] ThesisDataset (teacher signals, feature dict)
- [x] SFT trainer (3-stage loss)
- [x] RFT trainer (pairwise ranking RL)
- [x] Integration tests (16 passing)
- [x] Porting documentation (PORTING_SUMMARY.md)
- [x] Reference notebooks (RankingRL, SchemaValidation)

---

## Notes

- **Time Estimates:** Approximate; adjust based on team size and experience
- **Dependencies:** Some tasks can be parallelized; data integration can start immediately
- **Risk Areas:** LLM integration complexity, data quality/availability, production deployment
- **Success Criteria:** See README.md for detailed metrics and baselines

---

**Last Updated:** November 9, 2025
**Status:** Post-prototype, entering Research Alpha phase
