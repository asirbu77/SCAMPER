"""behavior_prior: Behavior-prior training pipeline.

Trains a behavior-prior network on top of a frozen distill_expert checkpoint
(encoder + decoder + motion prior). The encoder/decoder/motion-prior stay
frozen; only the behavior prior is trained, using KL divergence to match the
encoder's per-step latent distribution.
"""

from scamper.agent.behavior_prior.behavior_prior_train import train
from scamper.agent.behavior_prior.behavior_prior_networks import (
    load_frozen_distill_networks,
    make_behavior_prior_networks,
    create_combined_checkpoint_params,
)
from scamper.agent.behavior_prior.losses import (
    compute_behavior_prior_training_loss,
)
from scamper.agent.behavior_prior.behavior_prior_rollout_eval import (
    MultiModeBehaviorPriorRolloutEvaluator,
)

__all__ = [
    "train",
    "load_frozen_distill_networks",
    "make_behavior_prior_networks",
    "create_combined_checkpoint_params",
    "compute_behavior_prior_training_loss",
    "MultiModeBehaviorPriorRolloutEvaluator",
]
