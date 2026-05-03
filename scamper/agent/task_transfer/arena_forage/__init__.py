"""Arena forage task transfer training pipeline.

Four training modes:
1. mimic_mjx: Freeze decoder, train new encoder
2. res_control: Freeze motion prior + decoder, train residual encoder (from distill_expert checkpoint)
3. prior_gating: Freeze motion prior + behavior prior + decoder, train residual + gate (from behavior_prior checkpoint)
4. scratch: Train both policy and decoder from random initialization

Note: Do not add imports here that trigger mujoco imports.
The MUJOCO_GL environment variable must be set before mujoco is imported,
which happens in train.py. Adding imports here would cause mujoco to be
imported before train.py can set the environment variable.
"""

__all__ = [
    "DecoderHighLevelWrapper",
    "PriorDecoderHighLevelWrapper",
    "load_prior_checkpoint",
    "make_decoder_inference_fn",
    "make_prior_inference_fn",
]
