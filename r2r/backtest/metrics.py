"""Performance metrics calculation."""


import numpy as np

from r2r.utils.logging import get_logger

logger = get_logger(__name__)


class PerformanceMetrics:
    """Calculate backtesting performance metrics."""

    @staticmethod
    def calculate_accuracy(predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate classification accuracy.

        Args:
            predictions: Predicted labels
            actuals: Actual labels

        Returns:
            Accuracy (0-1)
        """
        return (predictions == actuals).mean()

    @staticmethod
    def calculate_returns(
        predictions: np.ndarray,
        actual_returns: np.ndarray,
        transaction_cost: float = 0.001,
    ) -> np.ndarray:
        """Calculate strategy returns.

        Args:
            predictions: Predicted labels (0-4)
            actual_returns: Actual returns
            transaction_cost: Transaction cost per trade

        Returns:
            Array of strategy returns
        """
        # Map predictions to positions
        # 0 (strong sell) -> -1, 1 (sell) -> -0.5, 2 (hold) -> 0,
        # 3 (buy) -> 0.5, 4 (strong buy) -> 1
        position_map = {0: -1.0, 1: -0.5, 2: 0.0, 3: 0.5, 4: 1.0}
        positions = np.array([position_map[p] for p in predictions])

        # Calculate strategy returns
        strategy_returns = positions * actual_returns

        # Apply transaction costs when position changes
        position_changes = np.diff(positions, prepend=0)
        costs = np.abs(position_changes) * transaction_cost
        strategy_returns -= costs

        return strategy_returns

    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate annualized Sharpe ratio.

        Args:
            returns: Daily returns
            risk_free_rate: Annual risk-free rate

        Returns:
            Sharpe ratio
        """
        daily_rf = risk_free_rate / 252  # Convert to daily
        excess_returns = returns - daily_rf

        if excess_returns.std() == 0:
            return 0.0

        sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        return sharpe

    @staticmethod
    def calculate_max_drawdown(cumulative_returns: np.ndarray) -> float:
        """Calculate maximum drawdown.

        Args:
            cumulative_returns: Cumulative return series

        Returns:
            Maximum drawdown (as positive percentage)
        """
        cumulative = np.cumprod(1 + cumulative_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_dd = abs(drawdown.min())
        return max_dd

    @staticmethod
    def calculate_win_rate(returns: np.ndarray) -> float:
        """Calculate win rate (percentage of positive returns).

        Args:
            returns: Strategy returns

        Returns:
            Win rate (0-1)
        """
        return (returns > 0).mean()

    @staticmethod
    def calculate_profit_factor(returns: np.ndarray) -> float:
        """Calculate profit factor (gross profit / gross loss).

        Args:
            returns: Strategy returns

        Returns:
            Profit factor
        """
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())

        if losses == 0:
            return float("inf") if gains > 0 else 0.0

        return gains / losses

    def calculate_all_metrics(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        actual_returns: np.ndarray,
        transaction_cost: float = 0.001,
    ) -> dict[str, float]:
        """Calculate all performance metrics.

        Args:
            predictions: Predicted labels
            actuals: Actual labels
            actual_returns: Actual returns
            transaction_cost: Transaction cost

        Returns:
            Dictionary of all metrics
        """
        # Classification metrics
        accuracy = self.calculate_accuracy(predictions, actuals)

        # Return-based metrics
        strategy_returns = self.calculate_returns(predictions, actual_returns, transaction_cost)
        total_return = (1 + strategy_returns).prod() - 1
        sharpe_ratio = self.calculate_sharpe_ratio(strategy_returns)
        max_drawdown = self.calculate_max_drawdown(strategy_returns)
        win_rate = self.calculate_win_rate(strategy_returns)
        profit_factor = self.calculate_profit_factor(strategy_returns)

        metrics = {
            "accuracy": accuracy,
            "total_return": total_return,
            "annualized_return": (1 + total_return) ** (252 / len(strategy_returns)) - 1,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "num_trades": (np.diff(predictions, prepend=predictions[0]) != 0).sum(),
        }

        logger.info(f"Calculated metrics: {metrics}")
        return metrics
