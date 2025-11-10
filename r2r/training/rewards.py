"""Reward calculation for RFT."""

from typing import Any, Optional

import numpy as np

from r2r.utils.logging import get_logger

logger = get_logger(__name__)


class RewardCalculator:
    """Calculate rewards for thesis quality and prediction accuracy."""

    def __init__(
        self,
        w_structure: float = 0.2,
        w_claims: float = 0.3,
        w_decision: float = 0.5,
    ):
        """Initialize reward calculator.

        Args:
            w_structure: Weight for structure quality reward
            w_claims: Weight for claims quality reward
            w_decision: Weight for decision accuracy reward
        """
        self.w_structure = w_structure
        self.w_claims = w_claims
        self.w_decision = w_decision

        # Validate weights sum to 1
        total = w_structure + w_claims + w_decision
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total}")

        logger.info(
            f"Initialized RewardCalculator: wS={w_structure}, " f"wC={w_claims}, wD={w_decision}"
        )

    def calculate_structure_reward(self, thesis: dict[str, Any]) -> float:
        """Calculate reward for thesis structure quality.

        Args:
            thesis: Thesis dictionary

        Returns:
            Structure reward (0-1)
        """
        reward = 0.0

        # Check completeness of sections
        required_sections = [
            "market_environment",
            "fundamental_strength",
            "sentiment_analysis",
            "technical_setup",
            "insider_activity",
            "risk_factors",
        ]

        if "thesis" in thesis:
            completed_sections = sum(
                1 for sec in required_sections if sec in thesis["thesis"] and thesis["thesis"][sec]
            )
            reward = completed_sections / len(required_sections)

        return reward

    def calculate_claims_reward(self, thesis: dict[str, Any]) -> float:
        """Calculate reward for claims quality.

        Args:
            thesis: Thesis dictionary

        Returns:
            Claims reward (0-1)
        """
        reward = 0.0

        if "evidence" not in thesis:
            return reward

        evidence = thesis["evidence"]

        # Reward for having evidence
        if len(evidence) > 0:
            reward += 0.3

            # Reward for evidence diversity (multiple sources)
            sources = set(item.get("source", "") for item in evidence)
            if len(sources) > 1:
                reward += 0.2

            # Reward for temporal information
            has_time = sum(1 for item in evidence if "timestamp" in item)
            reward += (has_time / len(evidence)) * 0.2

            # Reward for quote quality (length as proxy)
            avg_quote_len = np.mean([len(item.get("quote", "")) for item in evidence])
            if avg_quote_len > 50:  # Meaningful quotes
                reward += 0.3

        return min(reward, 1.0)

    def calculate_decision_reward(
        self,
        thesis: dict[str, Any],
        actual_return: float,
        horizon_days: int = 21,
    ) -> float:
        """Calculate reward for decision accuracy.

        Args:
            thesis: Thesis dictionary
            actual_return: Actual forward return realized
            horizon_days: Forecast horizon in days

        Returns:
            Decision reward (-1 to 1)
        """
        if "decision" not in thesis:
            return 0.0

        decision = thesis["decision"]
        predicted_label = decision.get("label", 2)  # Default to hold

        # Map return to label
        # This is simplified - should match label generation logic
        if actual_return < -0.05:
            actual_label = 0  # Strong sell
        elif actual_return < -0.01:
            actual_label = 1  # Sell
        elif actual_return < 0.01:
            actual_label = 2  # Hold
        elif actual_return < 0.05:
            actual_label = 3  # Buy
        else:
            actual_label = 4  # Strong buy

        # Calculate reward based on label distance
        label_distance = abs(predicted_label - actual_label)

        if label_distance == 0:
            reward = 1.0  # Perfect prediction
        elif label_distance == 1:
            reward = 0.5  # Close
        elif label_distance == 2:
            reward = 0.0  # Neutral
        elif label_distance == 3:
            reward = -0.5  # Wrong direction
        else:
            reward = -1.0  # Completely wrong

        # Additional reward for probability calibration
        probs = decision.get("probabilities", [0.2] * 5)
        predicted_prob = probs[predicted_label]
        _ = probs[actual_label]  # noqa: F841 - Reserved for future calibration metrics

        # Bonus for high confidence when correct
        if predicted_label == actual_label and predicted_prob > 0.5:
            reward += 0.2

        # Penalty for high confidence when wrong
        if predicted_label != actual_label and predicted_prob > 0.5:
            reward -= 0.2

        return np.clip(reward, -1.0, 1.0)

    def calculate_total_reward(
        self,
        thesis: dict[str, Any],
        actual_return: Optional[float] = None,
        horizon_days: int = 21,
    ) -> dict[str, float]:
        """Calculate total reward as weighted sum of components.

        Args:
            thesis: Thesis dictionary
            actual_return: Actual forward return (if available)
            horizon_days: Forecast horizon in days

        Returns:
            Dictionary with component rewards and total
        """
        r_structure = self.calculate_structure_reward(thesis)
        r_claims = self.calculate_claims_reward(thesis)

        # Decision reward requires actual return
        if actual_return is not None:
            r_decision = self.calculate_decision_reward(thesis, actual_return, horizon_days)
        else:
            r_decision = 0.0

        total_reward = (
            self.w_structure * r_structure + self.w_claims * r_claims + self.w_decision * r_decision
        )

        return {
            "r_structure": r_structure,
            "r_claims": r_claims,
            "r_decision": r_decision,
            "r_total": total_reward,
        }
