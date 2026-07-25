"""From-scratch PPO training of the SCAMPER three-network system.

Trains the same encoder + decoder + motion prior that ``train_distill_expert``
produces, but with PPO on the imitation reward instead of DAgger against a
frozen expert teacher. The two latent regularizers are kept: KL(encoder ||
motion prior) and the AR(1) temporal smoothness term on the encoder means.

Not to be confused with the ``scratch`` *mode* of ``agent.task_transfer``
(``task_transfer/<task>/scratch_networks.py``), which is a plain flat-obs MLP
policy with no latent structure.
"""
