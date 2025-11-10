"""Feature extraction utilities."""

from typing import Optional

import numpy as np
import pandas as pd

from r2r.utils.logging import get_logger

logger = get_logger(__name__)


class FeatureExtractor:
    """Extract features from financial data."""

    def __init__(self, feature_config: Optional[dict] = None):
        """Initialize feature extractor.

        Args:
            feature_config: Optional configuration for features
        """
        self.feature_config = feature_config or {}

    def extract_price_features(self, df: pd.DataFrame, price_cols: list[str]) -> pd.DataFrame:
        """Extract price-based features.

        Args:
            df: Input DataFrame
            price_cols: List of price column names

        Returns:
            DataFrame with price features added
        """
        df_out = df.copy()

        for col in price_cols:
            # Returns
            df_out[f"{col}_return_1d"] = df[col].pct_change(1)
            df_out[f"{col}_return_5d"] = df[col].pct_change(5)
            df_out[f"{col}_return_21d"] = df[col].pct_change(21)

            # Log returns
            df_out[f"{col}_log_return"] = np.log(df[col] / df[col].shift(1))

            # Volatility (rolling std of returns)
            df_out[f"{col}_volatility_21d"] = df_out[f"{col}_return_1d"].rolling(21).std()

        logger.info(f"Extracted price features for {len(price_cols)} columns")
        return df_out

    def extract_volume_features(self, df: pd.DataFrame, volume_cols: list[str]) -> pd.DataFrame:
        """Extract volume-based features.

        Args:
            df: Input DataFrame
            volume_cols: List of volume column names

        Returns:
            DataFrame with volume features added
        """
        df_out = df.copy()

        for col in volume_cols:
            # Volume changes
            df_out[f"{col}_change_1d"] = df[col].pct_change(1)

            # Volume moving averages
            df_out[f"{col}_ma_5d"] = df[col].rolling(5).mean()
            df_out[f"{col}_ma_21d"] = df[col].rolling(21).mean()

            # Volume ratio (current vs MA)
            df_out[f"{col}_ratio_ma21"] = df[col] / df_out[f"{col}_ma_21d"]

        logger.info(f"Extracted volume features for {len(volume_cols)} columns")
        return df_out

    def extract_momentum_features(self, df: pd.DataFrame, price_cols: list[str]) -> pd.DataFrame:
        """Extract momentum features.

        Args:
            df: Input DataFrame
            price_cols: List of price column names

        Returns:
            DataFrame with momentum features added
        """
        df_out = df.copy()

        for col in price_cols:
            # Rate of change
            df_out[f"{col}_roc_5d"] = (df[col] - df[col].shift(5)) / df[col].shift(5) * 100
            df_out[f"{col}_roc_21d"] = (df[col] - df[col].shift(21)) / df[col].shift(21) * 100

            # Momentum (price - price N days ago)
            df_out[f"{col}_momentum_5d"] = df[col] - df[col].shift(5)
            df_out[f"{col}_momentum_21d"] = df[col] - df[col].shift(21)

        logger.info(f"Extracted momentum features for {len(price_cols)} columns")
        return df_out

    def extract_all_features(
        self,
        df: pd.DataFrame,
        price_cols: list[str],
        volume_cols: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Extract all features.

        Args:
            df: Input DataFrame
            price_cols: List of price column names
            volume_cols: Optional list of volume column names

        Returns:
            DataFrame with all features
        """
        df_out = df.copy()

        # Price features
        df_out = self.extract_price_features(df_out, price_cols)

        # Momentum features
        df_out = self.extract_momentum_features(df_out, price_cols)

        # Volume features (if provided)
        if volume_cols:
            df_out = self.extract_volume_features(df_out, volume_cols)

        logger.info(f"Extracted all features: {df_out.shape}")
        return df_out
