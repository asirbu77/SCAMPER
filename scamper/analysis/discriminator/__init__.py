"""Discriminator training pipeline for motion clip classification.

This module provides a self-contained pipeline for training a binary classifier
to distinguish between different sources of motion clips (e.g., original reference
data vs prior-generated rollouts).
"""

from scamper.analysis.discriminator.data_loading import (
    MotionClipDataset,
    create_train_test_split,
    list_h5_datasets,
    load_h5_dataset,
    load_h5_metadata,
)
from scamper.analysis.discriminator.discriminator_network import (
    Discriminator,
    make_discriminator_network,
)
from scamper.analysis.discriminator.discriminator_train import (
    DiscriminatorParams,
    TrainingState,
    train,
)
from scamper.analysis.discriminator.rnn_discriminator_network import (
    AttentionPooling,
    RNNDiscriminator,
    make_rnn_discriminator_network,
)

__all__ = [
    "AttentionPooling",
    "Discriminator",
    "DiscriminatorParams",
    "MotionClipDataset",
    "RNNDiscriminator",
    "TrainingState",
    "create_train_test_split",
    "list_h5_datasets",
    "load_h5_dataset",
    "load_h5_metadata",
    "make_discriminator_network",
    "make_rnn_discriminator_network",
    "train",
]
