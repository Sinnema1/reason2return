"""Technical indicator calculations."""

from typing import Optional

import pandas as pd

from r2r.utils.logging import get_logger

logger = get_logger(__name__)


class TechnicalIndicators:
    """Calculate technical indicators for financial data."""

    @staticmethod
    def sma(series: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average.

        Args:
            series: Price series
            window: Window size

        Returns:
            SMA series
        """
        return series.rolling(window).mean()

    @staticmethod
    def ema(series: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average.

        Args:
            series: Price series
            window: Window size

        Returns:
            EMA series
        """
        return series.ewm(span=window, adjust=False).mean()

    @staticmethod
    def rsi(series: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index.

        Args:
            series: Price series
            window: Window size (default 14)

        Returns:
            RSI series (0-100)
        """
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Moving Average Convergence Divergence.

        Args:
            series: Price series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period

        Returns:
            DataFrame with MACD, signal, and histogram
        """
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return pd.DataFrame({"macd": macd_line, "signal": signal_line, "histogram": histogram})

    @staticmethod
    def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
        """Bollinger Bands.

        Args:
            series: Price series
            window: Window size
            num_std: Number of standard deviations

        Returns:
            DataFrame with upper, middle, and lower bands
        """
        middle = series.rolling(window).mean()
        std = series.rolling(window).std()

        upper = middle + (std * num_std)
        lower = middle - (std * num_std)

        return pd.DataFrame({"upper": upper, "middle": middle, "lower": lower})

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            window: Window size

        Returns:
            ATR series
        """
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window).mean()

        return atr

    def add_all_indicators(
        self, df: pd.DataFrame, price_col: str, prefix: Optional[str] = None
    ) -> pd.DataFrame:
        """Add all technical indicators to DataFrame.

        Args:
            df: Input DataFrame
            price_col: Name of price column
            prefix: Optional prefix for output columns

        Returns:
            DataFrame with indicators added
        """
        df_out = df.copy()
        pfx = f"{prefix}_" if prefix else ""

        # Moving averages
        df_out[f"{pfx}sma_20"] = self.sma(df[price_col], 20)
        df_out[f"{pfx}sma_50"] = self.sma(df[price_col], 50)
        df_out[f"{pfx}ema_12"] = self.ema(df[price_col], 12)
        df_out[f"{pfx}ema_26"] = self.ema(df[price_col], 26)

        # RSI
        df_out[f"{pfx}rsi_14"] = self.rsi(df[price_col], 14)

        # MACD
        macd_df = self.macd(df[price_col])
        df_out[f"{pfx}macd"] = macd_df["macd"]
        df_out[f"{pfx}macd_signal"] = macd_df["signal"]
        df_out[f"{pfx}macd_hist"] = macd_df["histogram"]

        # Bollinger Bands
        bb_df = self.bollinger_bands(df[price_col])
        df_out[f"{pfx}bb_upper"] = bb_df["upper"]
        df_out[f"{pfx}bb_middle"] = bb_df["middle"]
        df_out[f"{pfx}bb_lower"] = bb_df["lower"]

        logger.info(f"Added all technical indicators for {price_col}")
        return df_out
