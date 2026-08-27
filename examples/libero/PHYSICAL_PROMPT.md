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

## Effect-binding follow-up

The config `pi05_libero_effect_binding_lora` addresses the prompt-ignoring failure of ordinary behavior cloning. Each
prompt unit contains `(pre-image, action, post-image)`. A compact adapter gates the visual effect difference with the
aligned action before inserting one fused effect token. Training uses shared diffusion noise for the correct prompt,
an explicit wrong-task prompt, and the correct images with reversed actions. Hinge ranking requires both intervened
prompts to explain the target action worse than the correct prompt.

The three same-physics counterfactual task pairs qualified by the evaluator use explicit A/B prompt mappings. Other
tasks receive deterministic different-task negatives. This is still a mechanism experiment: evaluate with
`counterfactual_pair_eval.py --physical-prompt-effects` and require both prompt switching and a positive aligned-minus-
reversed action gap before scaling training or moving to LIBERO-Plus.

## Causal ICL curriculum

The config `pi05_libero_causal_icl_curriculum_lora` is the behavior-preserving C2 test. It restarts from the released
Pi0.5 checkpoint rather than inheriting the failed C1 weights. Four action-gated local change tokens retain the
strongest pooled `post-image - pre-image` patches per frame; C1's global mean-effect fusion remains available when
`physical_prompt_local_effect_tokens=0`.

Effect prompts use sparse pre-frame anchors distributed over the episode, but every post frame is exactly the next
environment frame. Thus each transition is `(image_i, action_i, image_{i+1})`; adjacent sparse anchors must never be
treated as an action effect because they contain unobserved intermediate actions.

The config `pi05_libero_causal_icl_aligned_effect_lora` is the C2.1 control: it is hyperparameter-identical to C2 but
has a separate experiment namespace so results from the corrected transition construction cannot overwrite the
original sparse-interval run.

The first 150 steps are behavior cloning only. A deterministic 25% of samples use the original task language with all
physical-prompt tokens masked, anchoring the released language-conditioned behavior. From step 150 to 299, wrong-task
and reversed-action rankings ramp linearly from zero to 0.15. Ranking is computed per example, normalized by the
matched-noise denoising-loss scale, and enabled only for the three evaluator-qualified same-physics task pairs. Keep
both step 150 and step 299: the former isolates the representation and behavior curriculum, while the latter measures
the causal ranking intervention.
