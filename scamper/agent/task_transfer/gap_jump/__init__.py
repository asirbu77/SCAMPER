"""Gap jump task transfer training pipeline.

Three training modes:
1. decoder_only: Freeze decoder, train new encoder
2. prior_decoder: Freeze prior + decoder, train residual encoder
3. scratch: Train both policy and decoder from random initialization

Note: Do not add imports here that trigger mujoco imports.
The MUJOCO_GL environment variable must be set before mujoco is imported,
which happens in train.py.
"""

__all__ = [
    "DecoderHighLevelWrapper",
    "PriorDecoderHighLevelWrapper",
    "load_prior_checkpoint",
    "make_decoder_inference_fn",
    "make_prior_inference_fn",
]
