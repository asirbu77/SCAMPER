"""Arena forage task transfer training pipeline.

Four training modes:
1. mimic_mjx: Freeze decoder, train new encoder
2. res_control: Freeze motion prior + decoder, train residual encoder (from distill_expert checkpoint)
3. prior_gating: Freeze motion prior + behavior prior + decoder, train residual + gate (from behavior_prior checkpoint)
4. scratch: Train both policy and decoder from random initialization

Do not add imports here that pull in mujoco. ``MUJOCO_GL`` must be set before
mujoco is first imported, which happens in train.py.
"""

__all__ = [
    "DecoderHighLevelWrapper",
    "PriorDecoderHighLevelWrapper",
    "load_prior_checkpoint",
    "make_decoder_inference_fn",
    "make_prior_inference_fn",
]
