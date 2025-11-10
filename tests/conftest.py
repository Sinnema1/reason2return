"""Pytest configuration and shared fixtures."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


@pytest.fixture(scope="session")
def project_root():
    """Return path to project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def config_dir(project_root):
    """Return path to configs directory."""
    return project_root / "configs"


@pytest.fixture(scope="session")
def schema_dir(project_root):
    """Return path to schemas directory."""
    return project_root / "schemas"


@pytest.fixture
def seed():
    """Fixed random seed for reproducible tests."""
    return 42


@pytest.fixture
def set_seed(seed):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@pytest.fixture
def synthetic_data(set_seed):
    """Generate small synthetic dataset for testing.

    Returns:
        Tuple of (df, X, y, feature_names)
    """
    num_days = 100
    tickers = ["AAPL", "MSFT", "GOOG"]

    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=num_days)
    rows = []

    for tic in tickers:
        # Simple random walks
        price = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, num_days)))
        f_quality = np.cumsum(np.random.normal(0, 0.02, num_days))
        f_sentiment = np.cumsum(np.random.normal(0, 0.03, num_days))

        for d in range(num_days):
            rows.append(
                {
                    "date": dates[d],
                    "ticker": tic,
                    "price": price[d],
                    "f_quality": f_quality[d],
                    "f_sentiment": f_sentiment[d],
                }
            )

    df = pd.DataFrame(rows).sort_values(["ticker", "date"]).reset_index(drop=True)
    df["ret1"] = df.groupby("ticker")["price"].pct_change().fillna(0.0)
    df["vol"] = (
        df.groupby("ticker")["ret1"].rolling(20).std().reset_index(level=0, drop=True).fillna(0.01)
    )

    # Create simple labels (3-class for simplicity)
    fwd = df.groupby("ticker")["price"].pct_change(5).shift(-5)
    z = fwd / (df["vol"] + 1e-8)
    df["label"] = pd.cut(z, bins=[-np.inf, -0.5, 0.5, np.inf], labels=[0, 1, 2]).astype(int)

    features = ["f_quality", "f_sentiment", "vol"]
    X = df[features].copy().fillna(0.0)
    y = df["label"].values

    return df, X, y, features


@pytest.fixture
def sample_thesis():
    """Sample valid thesis for testing."""
    return {
        "ticker": "AAPL",
        "as_of": "2025-11-09",
        "thesis": {
            "market": ["Market conditions favorable"],
            "fundamentals": ["Strong revenue growth"],
            "sentiment": ["Positive analyst coverage"],
            "technicals": ["Upward momentum"],
            "insider": ["No significant insider selling"],
            "risks": ["Macro headwinds"],
            "evidence": [
                {
                    "claim_id": "F1",
                    "quote": "Revenue up 12% YoY",
                    "source": "10-Q",
                    "time": "2025-10-30",
                }
            ],
        },
        "decision": {
            "label": "Buy",
            "probs": [0.05, 0.10, 0.20, 0.50, 0.15],
        },
    }


@pytest.fixture
def device():
    """Return appropriate torch device (CPU for M3 MacBook Air)."""
    # For Apple Silicon, we could use MPS but stick with CPU for consistency
    return torch.device("cpu")
