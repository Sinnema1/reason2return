"""Reinforcement fine-tuning trainer using pairwise ranking."""

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from r2r.training.rewards import RewardCalculator
from r2r.utils.logging import get_logger

logger = get_logger(__name__)


class ReinforceTrainer:
    """Trainer for reinforcement fine-tuning (RFT) using pairwise ranking.

    Implements the pairwise ranking RL approach from the notebooks:
    - Sample two theses per data point
    - Compute rewards R1, R2 for each
    - Update policy using advantage = 0.5 * (R1 - R2)
    - Loss = -(advantage1 * log_prob1 - advantage2 * log_prob2)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        reward_calculator: RewardCalculator,
        device: str = "cpu",
        entropy_coef: float = 0.01,
    ):
        """Initialize RFT trainer.

        Args:
            model: ThesisPipeline model to train
            optimizer: Optimizer
            reward_calculator: Reward calculator
            device: Device to train on
            entropy_coef: Coefficient for entropy regularization
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.reward_calculator = reward_calculator
        self.device = device
        self.entropy_coef = entropy_coef

        logger.info(
            f"Initialized ReinforceTrainer (pairwise ranking) on {device} "
            f"with entropy_coef={entropy_coef}"
        )

    def _build_thesis_from_samples(
        self, structure: torch.Tensor, claims: torch.Tensor, decision: torch.Tensor
    ) -> dict:
        """Build thesis dict from sampled outputs.

        Args:
            structure: Binary tensor (6,) for sections
            claims: Binary tensor (8,) for claims
            decision: Integer tensor for decision label

        Returns:
            Thesis dictionary matching schema
        """
        # Convert to numpy for dict building
        s = structure.cpu().numpy()
        c = claims.cpu().numpy()
        d = decision.cpu().item()

        # Map to schema structure
        section_names = self.model.SECTION_NAMES
        active_sections = [name for i, name in enumerate(section_names) if s[i] > 0.5]  # type: ignore[union-attr, index]

        # Build thesis
        thesis = {
            "structure": {name: (name in active_sections) for name in section_names},  # type: ignore[union-attr, index]
            "claims": {
                "financial_health": bool(c[0] > 0.5),
                "sentiment_positive": bool(c[1] > 0.5),
                "momentum_strong": bool(c[2] > 0.5),
                "insider_buying": bool(c[3] > 0.5),
                "high_volatility": bool(c[4] > 0.5),
                "short_term_momentum": bool(c[5] > 0.5),
                "long_term_momentum": bool(c[6] > 0.5),
                "sentiment_divergence": bool(c[7] > 0.5),
            },
            "decision": {
                "label": self.model.DECISION_LABELS[d],
                "confidence": 0.8,  # Could use softmax probs for this
            },
        }

        return thesis

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> dict[str, float]:
        """Train for one epoch using pairwise ranking.

        Args:
            train_loader: Training data loader (should be ThesisDataset)
            epoch: Current epoch number

        Returns:
            Epoch metrics
        """
        self.model.train()
        total_loss = 0.0
        total_reward = 0.0
        total_entropy = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"RFT Epoch {epoch}")
        for batch in pbar:
            # Unpack batch - need feature dict for reward calculation
            if len(batch) == 5:
                xb, yb, _, _, idx = batch
            else:
                xb, yb, _, _ = batch
                idx = None

            xb = xb.to(self.device)
            yb = yb.to(self.device)

            # Get feature dict for reward calculation
            if hasattr(train_loader.dataset, "get_feature_dict"):
                if idx is not None:
                    feature_dict = train_loader.dataset.get_feature_dict(idx)
                else:
                    # Fallback: use batch indices
                    batch_size = xb.size(0)
                    feature_dict = train_loader.dataset.get_feature_dict(torch.arange(batch_size))
            else:
                # Minimal fallback
                feature_dict = {"y": yb.cpu().numpy()}

            # Sample two theses per example
            (s1, c1, d1), (logp_s1, logp_c1, logp_d1), ent1 = self.model.sample(xb)  # type: ignore[operator]
            (s2, c2, d2), (logp_s2, logp_c2, logp_d2), ent2 = self.model.sample(xb)  # type: ignore[operator]

            # Compute total log probs (sum across outputs)
            log_prob1 = logp_s1.sum(dim=1) + logp_c1.sum(dim=1) + logp_d1
            log_prob2 = logp_s2.sum(dim=1) + logp_c2.sum(dim=1) + logp_d2

            # Generate theses for reward calculation
            theses1 = []
            theses2 = []
            for i in range(xb.size(0)):
                thesis1 = self._build_thesis_from_samples(s1[i], c1[i], d1[i])
                thesis2 = self._build_thesis_from_samples(s2[i], c2[i], d2[i])
                theses1.append(thesis1)
                theses2.append(thesis2)

            # Compute rewards
            R1 = self.reward_calculator.calculate_batch_rewards(theses1, feature_dict)  # type: ignore[attr-defined]
            R2 = self.reward_calculator.calculate_batch_rewards(theses2, feature_dict)  # type: ignore[attr-defined]

            # Convert to tensors
            R1 = torch.tensor(R1, device=self.device, dtype=torch.float32)
            R2 = torch.tensor(R2, device=self.device, dtype=torch.float32)

            # Pairwise advantage
            advantage1 = 0.5 * (R1 - R2)
            advantage2 = 0.5 * (R2 - R1)

            # REINFORCE loss with entropy bonus
            policy_loss = -(advantage1 * log_prob1 + advantage2 * log_prob2).mean()
            entropy_bonus = -(ent1 + ent2).mean() * self.entropy_coef
            total_loss_value = policy_loss + entropy_bonus

            # Backward pass
            self.optimizer.zero_grad()
            total_loss_value.backward()
            self.optimizer.step()

            # Track metrics
            total_loss += total_loss_value.item()
            total_reward += R1.mean().item()
            total_entropy += (ent1.mean().item() + ent2.mean().item()) / 2
            num_batches += 1

            pbar.set_postfix(
                {
                    "loss": total_loss / num_batches,
                    "reward": total_reward / num_batches,
                    "entropy": total_entropy / num_batches,
                }
            )

        avg_loss = total_loss / num_batches
        avg_reward = total_reward / num_batches
        avg_entropy = total_entropy / num_batches

        logger.info(
            f"RFT Epoch {epoch} - Loss: {avg_loss:.4f}, "
            f"Reward: {avg_reward:.4f}, Entropy: {avg_entropy:.4f}"
        )

        return {
            "loss": avg_loss,
            "reward": avg_reward,
            "entropy": avg_entropy,
        }

    def train(
        self,
        train_loader: DataLoader,
        num_epochs: int = 10,
        save_dir: Optional[Path] = None,
    ) -> dict[str, Any]:
        """Full RFT training loop.

        Args:
            train_loader: Training data loader
            num_epochs: Number of epochs
            save_dir: Optional save directory

        Returns:
            Training history
        """
        history: dict[str, list[float]] = {"loss": [], "reward": [], "entropy": []}

        best_reward = float("-inf")

        for epoch in range(1, num_epochs + 1):
            metrics = self.train_epoch(train_loader, epoch)

            history["loss"].append(metrics["loss"])
            history["reward"].append(metrics["reward"])
            history["entropy"].append(metrics["entropy"])

            # Save best model
            if save_dir and metrics["reward"] > best_reward:
                best_reward = metrics["reward"]
                save_path = save_dir / "best_rft_model.pt"
                self.save_checkpoint(save_path, epoch, metrics)
                logger.info(f"Saved best RFT model to {save_path}")

        logger.info("RFT training complete")
        return history

    def save_checkpoint(self, path: Path, epoch: int, metrics: dict[str, float]) -> None:
        """Save checkpoint.

        Args:
            path: Save path
            epoch: Current epoch
            metrics: Current metrics
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
            },
            path,
        )
