# Pi0.5 Physical Prompt baseline

This branch contains a deliberately minimal **sensorimotor in-context policy baseline**. It is an experimental
starting point, not evidence of new-task learning by itself.

## Input and model

Each live LIBERO observation is paired with eight uniformly sampled frames from another episode of the same task.
The language is fixed to `Follow the demonstrated behavior.` and therefore carries no task identity.

For every prompt timestep, the model receives:

- one external-camera RGB frame;
- the normalized robot action aligned to that frame.

SigLIP produces a 16x16 token map for every prompt frame. Spatial mean pooling reduces it to a 4x4 grid, so an
eight-frame prompt adds `8 * (16 visual + 1 action) + 2 boundary = 138` prefix tokens. Learned frame-position and
boundary embeddings distinguish the demonstration from the three live camera views. The released Pi0.5 backbone is
loaded unchanged; training updates only PaliGemma/action-expert LoRA parameters and four physical-prompt parameter
groups.

## Data access

The local LIBERO mirror is LeRobot v2 with one parquet file per episode and embedded PNGs. Physical-prompt training
uses a read-only row-group loader instead of asking Hugging Face Datasets to materialize a second Arrow copy. Episodes
and row-group blocks are shuffled, while frames within each block remain contiguous for NFS locality. The eight-frame
demonstration for a task is cached in process memory.

Required local resources:

```text
OPENPI_LEROBOT_ROOT=/datasets/L2_Simulation/LIBERO
OPENPI_PI05_LIBERO_CHECKPOINT=/share/longjunyu/pi05/cache/openpi/openpi-assets/checkpoints/pi05_libero
```

The training config is `pi05_libero_physical_prompt_lora`. On Core, use GPUs 0-2 and leave GPU3 retention running.
Write experiment checkpoints under `/share/longjunyu/pi05/checkpoints`, not `/workspace`.

## Required evaluation controls

A prompted checkpoint is not qualified by ordinary LIBERO success rate alone. Every same-scene task pair must use
matched initial simulator state and diffusion noise under these conditions:

1. correct sensorimotor prompt;
2. prompt from the counterfactual task;
3. correct prompt images with temporally shuffled or masked actions;
4. fully masked prompt;
5. released language-conditioned Pi0.5 baseline.

Primary metrics are paired success, prompt causal gap, and action-binding gap. A gain from correct versus wrong visual
prompt shows task selection; an additional gain from aligned versus shuffled actions is the minimum evidence that the
policy uses action-effect information.

## Scientific boundary

The public 40-task LIBERO checkpoint and dataset can validate the mechanism and the causal protocol, but most task
primitives are already in the checkpoint's training distribution. They cannot establish that the policy learned a new
primitive in context. That claim requires held-out rules or primitives, planned for LIBERO-Plus after the baseline
passes the same-scene causal gate.
