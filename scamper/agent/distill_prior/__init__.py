"""
distill_prior: Training pipeline for RGM network alignment to pretrained distill encoder.

This module trains an RGM (Random Generation Module) network to match the
encoder distributions from a pretrained distill checkpoint. The distill
encoder, decoder, and prior remain frozen; only the RGM is trained using
KL divergence loss.
"""

from scamper.agent.distill_prior.distill_prior_train import train
from scamper.agent.distill_prior.distill_prior_networks import (
    load_frozen_distill_networks,
    make_rgm_networks,
    create_combined_checkpoint_params,
)
from scamper.agent.distill_prior.losses import (
    compute_rgm_training_loss,
)
from scamper.agent.distill_prior.distill_prior_rollout_eval import (
    MultiModeRGMRolloutEvaluator,
)

__all__ = [
    "train",
    "load_frozen_distill_networks",
    "make_rgm_networks",
    "create_combined_checkpoint_params",
    "compute_rgm_training_loss",
    "MultiModeRGMRolloutEvaluator",
]
