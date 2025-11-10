"""R2R - Structured LLM Reasoning for Financial Decision Support.

This package implements a three-stage pipeline (Structure → Claims → Decision)
for generating structured financial theses with reinforcement learning.

Public API:
    - ThesisPipeline: Three-stage neural model
    - ThesisDataset: PyTorch dataset for training
    - SupervisedTrainer: SFT trainer
    - ReinforceTrainer: RFT trainer with pairwise ranking
"""

from r2r.data.dataset import ThesisDataset
from r2r.models.pipeline import ThesisPipeline
from r2r.training.rft import ReinforceTrainer
from r2r.training.sft import SupervisedTrainer

__version__ = "0.1.0"

__all__ = [
    "ThesisPipeline",
    "ThesisDataset",
    "SupervisedTrainer",
    "ReinforceTrainer",
]
