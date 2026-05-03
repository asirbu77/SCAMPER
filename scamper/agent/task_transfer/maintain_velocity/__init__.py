"""Maintain velocity task transfer training pipeline.

Three training modes:
1. mimic_mjx: Freeze decoder, train new encoder
2. res_control: Freeze motion prior + decoder, train residual encoder (from distill_expert checkpoint)
3. scratch: Train both policy and decoder from random initialization

Supports Orbax checkpointing and SLURM preemption recovery.

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
    "discover_existing_run_state",
    "save_run_state",
    "cleanup_run_state",
    "create_checkpoint_callback",
    "save_task_transfer_checkpoint",
    "load_task_transfer_metadata",
    "load_task_transfer_brax_params",
]
