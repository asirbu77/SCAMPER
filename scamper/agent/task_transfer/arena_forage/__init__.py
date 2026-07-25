"""Arena forage task transfer training pipeline.

Seven training modes:
1. mimic_mjx: Freeze decoder, train new encoder
2. res_control: Freeze motion prior + decoder, train residual encoder (from distill_expert checkpoint)
3. bp_res_control: Freeze behavior prior (rgm) + decoder, train residual encoder (from behavior_prior checkpoint)
4. prior_gating: Freeze motion prior + behavior prior + decoder, train residual + gate (from behavior_prior checkpoint)
5. prior_interp: Freeze motion prior + behavior prior + decoder, train residual with a fixed linear blend of both priors (from behavior_prior checkpoint)
6. prior_schedule: Like prior_gating, but the gate follows a fixed exploration schedule instead of being a policy output (from behavior_prior checkpoint)
7. scratch: Train both policy and decoder from random initialization

Do not add imports here that pull in mujoco. ``MUJOCO_GL`` must be set before
mujoco is first imported, which happens in train.py.
"""

__all__ = [
    "DecoderHighLevelWrapper",
    "PriorDecoderHighLevelWrapper",
    "PriorRGMDecoderHighLevelWrapper",
    "PriorRGMInterpDecoderHighLevelWrapper",
    "PriorScheduleDecoderHighLevelWrapper",
    "RGMDecoderHighLevelWrapper",
    "load_prior_checkpoint",
    "make_behavior_prior_inference_fn",
    "make_decoder_inference_fn",
    "make_prior_inference_fn",
]
