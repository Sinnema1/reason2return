"""Supervised fine-tuning trainer."""

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from r2r.utils.logging import get_logger

logger = get_logger(__name__)


class SupervisedTrainer:
    """Trainer for supervised fine-tuning (SFT)."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = "cpu",
    ):
        """Initialize SFT trainer.

        Args:
            model: Model to train
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        logger.info(f"Initialized SupervisedTrainer on {device}")

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> dict[str, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_struct_loss = 0.0
        total_claims_loss = 0.0
        total_decision_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # Unpack batch
            if len(batch) == 5:
                xb, yb, teacher_struct, teacher_claims, _ = batch
            else:
                xb, yb, teacher_struct, teacher_claims = batch

            # Move to device
            xb = xb.to(self.device)
            yb = yb.to(self.device)
            teacher_struct = teacher_struct.to(self.device)
            teacher_claims = teacher_claims.to(self.device)

            # Forward pass
            outputs = self.model(xb)

            # Compute losses for each stage
            struct_loss = nn.functional.binary_cross_entropy_with_logits(
                outputs["structure_logits"], teacher_struct
            )

            claims_loss = nn.functional.binary_cross_entropy_with_logits(
                outputs["claims_logits"], teacher_claims
            )

            decision_loss = self.criterion(outputs["decision_logits"], yb)

            # Combined loss
            loss = struct_loss + claims_loss + decision_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            total_struct_loss += struct_loss.item()
            total_claims_loss += claims_loss.item()
            total_decision_loss += decision_loss.item()
            num_batches += 1

            pbar.set_postfix(
                {
                    "loss": total_loss / num_batches,
                    "struct": total_struct_loss / num_batches,
                    "claims": total_claims_loss / num_batches,
                    "dec": total_decision_loss / num_batches,
                }
            )

        avg_loss = total_loss / num_batches
        logger.info(
            f"Epoch {epoch} - Loss: {avg_loss:.4f} "
            f"(struct={total_struct_loss/num_batches:.4f}, "
            f"claims={total_claims_loss/num_batches:.4f}, "
            f"decision={total_decision_loss/num_batches:.4f})"
        )

        return {
            "loss": avg_loss,
            "struct_loss": total_struct_loss / num_batches,
            "claims_loss": total_claims_loss / num_batches,
            "decision_loss": total_decision_loss / num_batches,
        }

    def evaluate(self, eval_loader: DataLoader) -> dict[str, float]:
        """Evaluate model on validation set.

        Args:
            eval_loader: Evaluation data loader

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].to(self.device)

                outputs = self.model(**inputs)
                loss = self.criterion(outputs["decision_logits"], labels)

                total_loss += loss.item()

                # Calculate accuracy
                predictions = outputs["decision_logits"].argmax(dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(eval_loader)
        accuracy = correct / total

        logger.info(f"Validation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        return {"loss": avg_loss, "accuracy": accuracy}

    def train(
        self,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        num_epochs: int = 10,
        save_dir: Optional[Path] = None,
    ) -> dict[str, Any]:
        """Full training loop.

        Args:
            train_loader: Training data loader
            eval_loader: Optional evaluation data loader
            num_epochs: Number of epochs to train
            save_dir: Optional directory to save checkpoints

        Returns:
            Training history
        """
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_accuracy": []}

        best_val_loss = float("inf")

        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            history["train_loss"].append(train_metrics["loss"])

            # Evaluate
            if eval_loader is not None:
                eval_metrics = self.evaluate(eval_loader)
                history["val_loss"].append(eval_metrics["loss"])
                history["val_accuracy"].append(eval_metrics["accuracy"])

                # Save best model
                if save_dir and eval_metrics["loss"] < best_val_loss:
                    best_val_loss = eval_metrics["loss"]
                    save_path = save_dir / "best_model.pt"
                    self.save_checkpoint(save_path, epoch, eval_metrics)
                    logger.info(f"Saved best model to {save_path}")

        logger.info("Training complete")
        return history

    def save_checkpoint(self, path: Path, epoch: int, metrics: dict[str, float]) -> None:
        """Save training checkpoint.

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

    def load_checkpoint(self, path: Path) -> dict[str, Any]:
        """Load training checkpoint.

        Args:
            path: Checkpoint path

        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint
