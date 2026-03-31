"""DAgger-based distillation from imitation_mlp teacher to VAE student.

Distills a trained flat MLP teacher into a VAE student (encoder + decoder +
jointly trained prior) using online distillation (DAgger). The student rolls
out in the environment, the teacher annotates visited states with correct
actions, and the student is trained via supervised learning through a
variational information bottleneck.
"""
