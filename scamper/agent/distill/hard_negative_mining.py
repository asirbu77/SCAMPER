"""Hard-negative mining for DAgger distillation.

Tracks which motion clips cause early termination during training and
provides weighted clip indices for oversampling difficult clips.

Operates outside jit/pmap at Python level during evaluation boundaries.
"""

import logging
from typing import Optional

import numpy as np


class HardNegativeMiner:
    """Track early terminations per clip and provide weighted clip sets.

    Uses exponential moving average (EMA) of termination rates to determine
    clip difficulty. Difficult clips are oversampled by duplicating their
    indices in the clip list.

    Args:
        num_clips: Total number of clips in the dataset.
        alpha: EMA smoothing factor for termination rate updates.
        max_weight: Maximum oversampling factor for any single clip.
    """

    def __init__(
        self,
        num_clips: int,
        alpha: float = 0.1,
        max_weight: float = 5.0,
    ):
        self.num_clips = num_clips
        self.alpha = alpha
        self.max_weight = max_weight

        # EMA of termination rate per clip (1.0 = always terminates early)
        self.termination_rates = np.zeros(num_clips)
        # Count of episodes seen per clip (for warming up the EMA)
        self.episode_counts = np.zeros(num_clips, dtype=np.int64)

    def update_from_eval(
        self,
        clip_indices: np.ndarray,
        early_terminations: np.ndarray,
    ) -> None:
        """Update termination statistics from evaluation rollouts.

        Args:
            clip_indices: Array of clip indices from evaluation episodes.
                Shape [num_episodes].
            early_terminations: Binary array, 1 if episode terminated early.
                Shape [num_episodes].
        """
        clip_indices = np.asarray(clip_indices).flatten()
        early_terminations = np.asarray(early_terminations).flatten()

        for clip_idx, terminated in zip(clip_indices, early_terminations):
            if clip_idx < 0 or clip_idx >= self.num_clips:
                continue
            self.episode_counts[clip_idx] += 1
            # EMA update
            self.termination_rates[clip_idx] = (
                1 - self.alpha
            ) * self.termination_rates[clip_idx] + self.alpha * float(terminated)

    def get_clip_weights(self) -> np.ndarray:
        """Return sampling weights proportional to difficulty.

        Clips with higher termination rates get higher weights.
        Clips that have never been seen get weight 1.0.

        Returns:
            Array of weights, shape [num_clips].
        """
        # Only use clips that have been seen at least once
        seen_mask = self.episode_counts > 0
        if not np.any(seen_mask):
            return np.ones(self.num_clips)

        weights = np.ones(self.num_clips)
        # Scale weights by termination rate
        # A clip with termination_rate=1.0 gets max_weight
        # A clip with termination_rate=0.0 gets weight 1.0
        weights[seen_mask] = (
            1.0 + (self.max_weight - 1.0) * self.termination_rates[seen_mask]
        )
        return weights

    def create_weighted_clip_indices(self) -> Optional[list[int]]:
        """Create a list of clip indices with difficult clips duplicated.

        Returns:
            List of clip indices where difficult clips appear multiple times,
            or None if no oversampling is needed yet.
        """
        weights = self.get_clip_weights()

        # Normalize weights so mean is 1.0
        mean_weight = weights.mean()
        if mean_weight <= 0:
            return None

        normalized = weights / mean_weight

        # Round to integer counts (minimum 1 per clip)
        counts = np.maximum(np.round(normalized).astype(int), 1)

        # Build index list with duplicates
        indices = []
        for clip_idx, count in enumerate(counts):
            indices.extend([clip_idx] * count)

        # Only return if there's actual oversampling happening
        if len(indices) == self.num_clips:
            return None

        logging.info(
            f"Hard-negative mining: {self.num_clips} clips -> {len(indices)} "
            f"weighted clips (max termination rate: {self.termination_rates.max():.3f})"
        )
        return indices

    def get_stats(self) -> dict:
        """Return summary statistics for logging.

        Returns:
            Dict with mining statistics.
        """
        seen_mask = self.episode_counts > 0
        n_seen = np.sum(seen_mask)

        stats = {
            "mining/clips_seen": int(n_seen),
            "mining/clips_total": self.num_clips,
        }

        if n_seen > 0:
            seen_rates = self.termination_rates[seen_mask]
            stats.update(
                {
                    "mining/mean_termination_rate": float(seen_rates.mean()),
                    "mining/max_termination_rate": float(seen_rates.max()),
                    "mining/min_termination_rate": float(seen_rates.min()),
                    "mining/num_hard_clips": int(np.sum(seen_rates > 0.5)),
                }
            )

        return stats
