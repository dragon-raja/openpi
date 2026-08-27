import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    # Number of sparse video frames used as a sensorimotor physical prompt.
    # Zero keeps the released pi0/pi0.5 architecture and checkpoint structure
    # unchanged.
    physical_prompt_frames: int = 0
    # Spatial grid retained from each 16x16 SigLIP token map.  A 4x4 grid turns
    # each prompt frame into 16 tokens instead of 256.
    physical_prompt_pool_grid: int = 4
    # Encode action effects from explicit (pre-image, action, post-image)
    # transition units rather than independent frame-action pairs.
    physical_prompt_effects: bool = False
    # Temporal scales k for exactly covered (o_t, a_t:t+k, o_t+k)
    # transitions. Empty preserves the single-action C2 representation.
    physical_prompt_effect_horizons: tuple[int, ...] = ()
    # Number of largest local post-minus-pre changes retained as separate
    # action-gated tokens per prompt frame. Zero preserves the C1 global
    # mean-effect fusion path.
    physical_prompt_local_effect_tokens: int = 0
    # Width of the explicit behavior-binding bottleneck. Zero disables the C3
    # behavior token and retrieval objective.
    physical_prompt_behavior_latent_dim: int = 0
    # Add an explicit mean adjacent-action difference. Unlike positional mean
    # pooling, this feature changes direction under a padding-safe reversal.
    physical_prompt_directed_action_flow: bool = False
    # Bind the visual post-minus-pre effect to the aligned action through a
    # multiplicative gate before forming the behavior latent.
    physical_prompt_cross_modal_behavior_binding: bool = False
    # Condition the query behavior on the current base-camera observation and
    # robot state, in addition to the supervised action chunk.
    physical_prompt_query_context_binding: bool = False
    # Keep one behavior latent per demonstration transition and align a local
    # query to the best compatible stage instead of averaging the full video.
    physical_prompt_stage_alignment: bool = False
    # Route to a demonstration stage using pre-action visual state only, then
    # verify action-effect behavior under that candidate-invariant route.
    physical_prompt_visual_stage_routing: bool = False
    # Carry an explicit wrong-task demonstration during training.  These
    # fields are ignored by inference and only materialized by the causal
    # ranking objective.
    physical_prompt_counterfactuals: bool = False

    pytorch_compile_mode: str | None = "max-autotune"

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.physical_prompt_frames < 0:
            raise ValueError("physical_prompt_frames must be non-negative")
        if self.physical_prompt_pool_grid <= 0 or 16 % self.physical_prompt_pool_grid != 0:
            raise ValueError("physical_prompt_pool_grid must be a positive divisor of 16")
        if self.physical_prompt_effects and not self.physical_prompt_frames:
            raise ValueError("physical_prompt_effects requires physical_prompt_frames > 0")
        if self.physical_prompt_effect_horizons and not self.physical_prompt_effects:
            raise ValueError("physical_prompt_effect_horizons requires physical_prompt_effects")
        if any(horizon <= 0 for horizon in self.physical_prompt_effect_horizons):
            raise ValueError("physical_prompt_effect_horizons must be positive")
        if self.physical_prompt_effect_horizons and max(self.physical_prompt_effect_horizons) > self.action_horizon:
            raise ValueError("physical_prompt_effect_horizons cannot exceed action_horizon")
        if self.physical_prompt_local_effect_tokens < 0:
            raise ValueError("physical_prompt_local_effect_tokens must be non-negative")
        if self.physical_prompt_local_effect_tokens and not self.physical_prompt_effects:
            raise ValueError("physical_prompt_local_effect_tokens requires physical_prompt_effects")
        if self.physical_prompt_local_effect_tokens > self.physical_prompt_pool_grid**2:
            raise ValueError("physical_prompt_local_effect_tokens cannot exceed the pooled visual token count")
        if self.physical_prompt_behavior_latent_dim < 0:
            raise ValueError("physical_prompt_behavior_latent_dim must be non-negative")
        if self.physical_prompt_behavior_latent_dim and not self.physical_prompt_effect_horizons:
            raise ValueError("physical_prompt_behavior_latent_dim requires multi-step physical-prompt effects")
        if self.physical_prompt_directed_action_flow and not self.physical_prompt_behavior_latent_dim:
            raise ValueError("physical_prompt_directed_action_flow requires a behavior latent")
        if self.physical_prompt_cross_modal_behavior_binding and not self.physical_prompt_behavior_latent_dim:
            raise ValueError("physical_prompt_cross_modal_behavior_binding requires a behavior latent")
        if self.physical_prompt_query_context_binding and not self.physical_prompt_behavior_latent_dim:
            raise ValueError("physical_prompt_query_context_binding requires a behavior latent")
        if self.physical_prompt_stage_alignment and not self.physical_prompt_behavior_latent_dim:
            raise ValueError("physical_prompt_stage_alignment requires a behavior latent")
        if self.physical_prompt_visual_stage_routing and not self.physical_prompt_stage_alignment:
            raise ValueError("physical_prompt_visual_stage_routing requires stage alignment")
        if self.physical_prompt_counterfactuals and not self.physical_prompt_frames:
            raise ValueError("physical_prompt_counterfactuals requires physical_prompt_frames > 0")
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    **{f"physical_prompt_{frame:03d}_rgb": image_spec for frame in range(self.physical_prompt_frames)},
                    **(
                        {
                            f"physical_prompt_post_{frame:03d}_rgb": image_spec
                            for frame in range(self.physical_prompt_frames)
                        }
                        if self.physical_prompt_effects
                        else {}
                    ),
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    **{
                        f"physical_prompt_{frame:03d}_rgb": image_mask_spec
                        for frame in range(self.physical_prompt_frames)
                    },
                    **(
                        {
                            f"physical_prompt_post_{frame:03d}_rgb": image_mask_spec
                            for frame in range(self.physical_prompt_frames)
                        }
                        if self.physical_prompt_effects
                        else {}
                    ),
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                physical_prompt_actions=(
                    jax.ShapeDtypeStruct(
                        (
                            [
                                batch_size,
                                self.physical_prompt_frames,
                                max(self.physical_prompt_effect_horizons),
                                self.action_dim,
                            ]
                            if self.physical_prompt_effect_horizons
                            else [batch_size, self.physical_prompt_frames, self.action_dim]
                        ),
                        jnp.float32,
                    )
                    if self.physical_prompt_frames
                    else None
                ),
                physical_prompt_action_mask=(
                    jax.ShapeDtypeStruct(
                        (
                            [batch_size, self.physical_prompt_frames, max(self.physical_prompt_effect_horizons)]
                            if self.physical_prompt_effect_horizons
                            else [batch_size, self.physical_prompt_frames]
                        ),
                        bool,
                    )
                    if self.physical_prompt_frames
                    else None
                ),
                physical_prompt_counterfactual_images=(
                    {
                        **{
                            f"physical_prompt_{frame:03d}_rgb": image_spec
                            for frame in range(self.physical_prompt_frames)
                        },
                        **(
                            {
                                f"physical_prompt_post_{frame:03d}_rgb": image_spec
                                for frame in range(self.physical_prompt_frames)
                            }
                            if self.physical_prompt_effects
                            else {}
                        ),
                    }
                    if self.physical_prompt_counterfactuals
                    else None
                ),
                physical_prompt_counterfactual_image_masks=(
                    {
                        **{
                            f"physical_prompt_{frame:03d}_rgb": image_mask_spec
                            for frame in range(self.physical_prompt_frames)
                        },
                        **(
                            {
                                f"physical_prompt_post_{frame:03d}_rgb": image_mask_spec
                                for frame in range(self.physical_prompt_frames)
                            }
                            if self.physical_prompt_effects
                            else {}
                        ),
                    }
                    if self.physical_prompt_counterfactuals
                    else None
                ),
                physical_prompt_counterfactual_actions=(
                    jax.ShapeDtypeStruct(
                        (
                            [
                                batch_size,
                                self.physical_prompt_frames,
                                max(self.physical_prompt_effect_horizons),
                                self.action_dim,
                            ]
                            if self.physical_prompt_effect_horizons
                            else [batch_size, self.physical_prompt_frames, self.action_dim]
                        ),
                        jnp.float32,
                    )
                    if self.physical_prompt_counterfactuals
                    else None
                ),
                physical_prompt_counterfactual_action_mask=(
                    jax.ShapeDtypeStruct(
                        (
                            [batch_size, self.physical_prompt_frames, max(self.physical_prompt_effect_horizons)]
                            if self.physical_prompt_effect_horizons
                            else [batch_size, self.physical_prompt_frames]
                        ),
                        bool,
                    )
                    if self.physical_prompt_counterfactuals
                    else None
                ),
                physical_prompt_rank_mask=(
                    jax.ShapeDtypeStruct([batch_size], bool) if self.physical_prompt_counterfactuals else None
                ),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)

    def get_physical_prompt_freeze_filter(self) -> nnx.filterlib.Filter:
        """Freeze everything except LoRA and physical-prompt parameters."""
        if not self.physical_prompt_frames:
            raise ValueError("Physical-prompt freezing requires physical_prompt_frames > 0")
        return nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*(lora|physical_prompt).*")),
        )
