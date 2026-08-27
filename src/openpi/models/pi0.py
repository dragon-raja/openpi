import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def _safe_l2_normalize(value, *, epsilon: float = 1e-6):
    """Normalize without the undefined zero-vector gradient of ``linalg.norm``."""
    squared_norm = jnp.sum(jnp.square(value), axis=-1, keepdims=True)
    return value * jax.lax.rsqrt(squared_norm + epsilon)


def _masked_action_flow(action_tokens, action_mask):
    """Mean directed adjacent difference over the valid action prefix."""
    pair_mask = jnp.logical_and(action_mask[:, 1:], action_mask[:, :-1])
    differences = action_tokens[:, 1:] - action_tokens[:, :-1]
    mask = pair_mask.astype(differences.dtype)
    denominator = jnp.maximum(jnp.sum(mask, axis=1, keepdims=True), 1.0)
    return jnp.sum(differences * mask[..., None], axis=1) / denominator


def _masked_logmeanexp_similarity(query, candidates, candidate_mask, *, temperature: float):
    """Smoothly align each query to a variable-size set of candidate stages."""
    similarities = jnp.einsum("bd,bpd->bp", query, candidates)
    mask = candidate_mask.astype(bool)
    valid_count = jnp.maximum(jnp.sum(mask, axis=-1), 1)
    masked_logits = jnp.where(mask, similarities / temperature, -1.0e4)
    scores = temperature * (jax.nn.logsumexp(masked_logits, axis=-1) - jnp.log(valid_count))
    return jnp.where(jnp.any(mask, axis=-1), scores, 0.0)


def _masked_routed_similarity(query, candidates, candidate_mask, route_query, route_keys, *, temperature: float):
    """Route with state-only keys, then score behavior without candidate-dependent selection."""
    mask = candidate_mask.astype(bool)
    route_logits = jnp.einsum("bd,bpd->bp", route_query, route_keys) / temperature
    route_logits = jnp.where(mask, route_logits, -1.0e4)
    route_weights = jax.nn.softmax(route_logits, axis=-1)
    route_weights = jnp.where(mask, route_weights, 0.0)
    route_weights /= jnp.maximum(jnp.sum(route_weights, axis=-1, keepdims=True), 1.0e-6)
    behavior_similarity = jnp.einsum("bd,bpd->bp", query, candidates)
    scores = jnp.sum(route_weights * behavior_similarity, axis=-1)
    return jnp.where(jnp.any(mask, axis=-1), scores, 0.0), route_weights


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.physical_prompt_frames = config.physical_prompt_frames
        self.physical_prompt_pool_grid = config.physical_prompt_pool_grid
        self.physical_prompt_effects = config.physical_prompt_effects
        self.physical_prompt_effect_horizons = config.physical_prompt_effect_horizons
        self.physical_prompt_local_effect_tokens = config.physical_prompt_local_effect_tokens
        self.physical_prompt_behavior_latent_dim = config.physical_prompt_behavior_latent_dim
        self.physical_prompt_directed_action_flow = config.physical_prompt_directed_action_flow
        self.physical_prompt_cross_modal_behavior_binding = config.physical_prompt_cross_modal_behavior_binding
        self.physical_prompt_query_context_binding = config.physical_prompt_query_context_binding
        self.physical_prompt_stage_alignment = config.physical_prompt_stage_alignment
        self.physical_prompt_visual_stage_routing = config.physical_prompt_visual_stage_routing
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.physical_prompt_frames:
            # Prompt actions are prefix context, so they must use the PaliGemma
            # width rather than the (smaller) action-expert width.
            self.physical_prompt_action_in_proj = nnx.Linear(config.action_dim, paligemma_config.width, rngs=rngs)
            self.physical_prompt_position_embedding = nnx.Param(
                jax.random.normal(
                    rngs.params(),
                    (config.physical_prompt_frames, paligemma_config.width),
                    dtype=jnp.float32,
                )
                * 0.02
            )
            self.physical_prompt_boundary_embedding = nnx.Param(
                jax.random.normal(rngs.params(), (2, paligemma_config.width), dtype=jnp.float32) * 0.02
            )
            if config.physical_prompt_effects:
                effect_width = paligemma_config.width // 8
                self.physical_prompt_effect_down = nnx.Linear(paligemma_config.width, effect_width, rngs=rngs)
                self.physical_prompt_effect_gate = nnx.Linear(paligemma_config.width, effect_width, rngs=rngs)
                self.physical_prompt_effect_up = nnx.Linear(effect_width, paligemma_config.width, rngs=rngs)
                if config.physical_prompt_local_effect_tokens:
                    self.physical_prompt_effect_token_embedding = nnx.Param(
                        jax.random.normal(
                            rngs.params(),
                            (config.physical_prompt_local_effect_tokens, paligemma_config.width),
                            dtype=jnp.float32,
                        )
                        * 0.02
                    )
            if config.physical_prompt_effect_horizons:
                max_effect_horizon = max(config.physical_prompt_effect_horizons)
                temporal_width = paligemma_config.width // 8
                self.physical_prompt_action_temporal_position_embedding = nnx.Param(
                    jax.random.normal(rngs.params(), (max_effect_horizon, paligemma_config.width), dtype=jnp.float32)
                    * 0.02
                )
                self.physical_prompt_query_action_position_embedding = nnx.Param(
                    jax.random.normal(rngs.params(), (config.action_horizon, paligemma_config.width), dtype=jnp.float32)
                    * 0.02
                )
                self.physical_prompt_action_temporal_down = nnx.Linear(
                    paligemma_config.width, temporal_width, rngs=rngs
                )
                self.physical_prompt_action_temporal_up = nnx.Linear(temporal_width, paligemma_config.width, rngs=rngs)
                if config.physical_prompt_directed_action_flow:
                    self.physical_prompt_action_flow_down = nnx.Linear(
                        paligemma_config.width, temporal_width, rngs=rngs
                    )
                    self.physical_prompt_action_flow_up = nnx.Linear(temporal_width, paligemma_config.width, rngs=rngs)
            if config.physical_prompt_behavior_latent_dim:
                self.physical_prompt_behavior_down = nnx.Linear(
                    paligemma_config.width, config.physical_prompt_behavior_latent_dim, rngs=rngs
                )
                self.physical_prompt_behavior_up = nnx.Linear(
                    config.physical_prompt_behavior_latent_dim, paligemma_config.width, rngs=rngs
                )
                self.physical_prompt_behavior_token_embedding = nnx.Param(
                    jax.random.normal(rngs.params(), (paligemma_config.width,), dtype=jnp.float32) * 0.02
                )
                if config.physical_prompt_query_context_binding:
                    self.physical_prompt_query_state_proj = nnx.Linear(
                        config.action_dim, paligemma_config.width, rngs=rngs
                    )
                if config.physical_prompt_visual_stage_routing:
                    self.physical_prompt_stage_route_down = nnx.Linear(
                        paligemma_config.width, config.physical_prompt_behavior_latent_dim, rngs=rngs
                    )
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    def _pool_physical_prompt_image(self, obs: _model.Observation, name: str):
        image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
        side = int(image_tokens.shape[1] ** 0.5)
        if side * side != image_tokens.shape[1] or side % self.physical_prompt_pool_grid:
            raise ValueError(
                "Physical-prompt pooling requires a square vision token map divisible by "
                f"{self.physical_prompt_pool_grid}; got {image_tokens.shape[1]} tokens"
            )
        pool = side // self.physical_prompt_pool_grid
        return einops.reduce(
            image_tokens,
            "b (gh ph gw pw) d -> b (gh gw) d",
            "mean",
            gh=self.physical_prompt_pool_grid,
            gw=self.physical_prompt_pool_grid,
            ph=pool,
            pw=pool,
        )

    def _encode_action_sequence(self, actions, action_mask, position_embedding):
        action_tokens = self.physical_prompt_action_in_proj(actions)
        positions = position_embedding[: actions.shape[1]]
        temporal = jax.nn.gelu(self.physical_prompt_action_temporal_down(action_tokens + positions[None, :, :]))
        temporal = self.physical_prompt_action_temporal_up(temporal)
        mask = action_mask.astype(temporal.dtype)
        denominator = jnp.maximum(jnp.sum(mask, axis=1, keepdims=True), 1.0)
        action_token = jnp.sum(temporal * mask[..., None], axis=1) / denominator
        if self.physical_prompt_directed_action_flow:
            directed_flow = _masked_action_flow(action_tokens, action_mask)
            directed_flow = jax.nn.gelu(self.physical_prompt_action_flow_down(directed_flow))
            action_token += self.physical_prompt_action_flow_up(directed_flow)
        return action_token

    def _encode_physical_prompt_action(self, actions, action_mask):
        if actions.ndim == 2:
            return self.physical_prompt_action_in_proj(actions), action_mask
        if actions.ndim != 3 or action_mask.ndim != 2:
            raise ValueError(
                "Multi-step prompt actions and masks must have shapes [batch, horizon, action] and [batch, horizon]"
            )
        action_token = self._encode_action_sequence(
            actions,
            action_mask,
            self.physical_prompt_action_temporal_position_embedding.value,
        )
        return action_token, jnp.any(action_mask, axis=1)

    def _behavior_frame_representation(self, pre_tokens, post_tokens, action_token):
        visual_summary = jnp.mean(pre_tokens, axis=1)
        effect_summary = jnp.mean(post_tokens - pre_tokens, axis=1)
        if self.physical_prompt_cross_modal_behavior_binding:
            effect_latent = jax.nn.gelu(self.physical_prompt_effect_down(effect_summary))
            action_gate = 2.0 * jax.nn.sigmoid(self.physical_prompt_effect_gate(action_token))
            effect_summary = self.physical_prompt_effect_up(effect_latent * action_gate)
        return action_token + visual_summary + effect_summary

    def _pool_behavior_frames(self, frame_representations, frame_masks):
        frames = jnp.stack(frame_representations, axis=1)
        masks = jnp.stack(frame_masks, axis=1)
        mask = masks.astype(frames.dtype)
        denominator = jnp.maximum(jnp.sum(mask, axis=1, keepdims=True), 1.0)
        pooled = jnp.sum(frames * mask[..., None], axis=1) / denominator
        latent = self.physical_prompt_behavior_down(pooled)
        return _safe_l2_normalize(latent)

    def _encode_physical_prompt_behavior_frames(self, obs: _model.Observation):
        prompt_names = sorted(
            name
            for name in obs.images
            if name.startswith("physical_prompt_") and not name.startswith("physical_prompt_post_")
        )
        prompt_post_names = sorted(name for name in obs.images if name.startswith("physical_prompt_post_"))
        if len(prompt_names) != self.physical_prompt_frames or len(prompt_post_names) != self.physical_prompt_frames:
            raise ValueError("Behavior binding requires one pre/post image pair per physical-prompt frame")
        frame_representations = []
        frame_route_representations = []
        frame_masks = []
        frame_order_masks = []
        for frame, name in enumerate(prompt_names):
            pre_tokens = self._pool_physical_prompt_image(obs, name)
            post_name = prompt_post_names[frame]
            post_tokens = self._pool_physical_prompt_image(obs, post_name)
            action_token, action_valid = self._encode_physical_prompt_action(
                obs.physical_prompt_actions[:, frame], obs.physical_prompt_action_mask[:, frame]
            )
            effect_mask = jnp.logical_and(action_valid, obs.image_masks[name])
            effect_mask = jnp.logical_and(effect_mask, obs.image_masks[post_name])
            frame_representations.append(self._behavior_frame_representation(pre_tokens, post_tokens, action_token))
            frame_route_representations.append(jnp.mean(pre_tokens, axis=1))
            frame_masks.append(effect_mask)
            order_valid = jnp.sum(obs.physical_prompt_action_mask[:, frame], axis=-1) >= 2
            frame_order_masks.append(jnp.logical_and(effect_mask, order_valid))
        return frame_representations, frame_route_representations, frame_masks, frame_order_masks

    def encode_physical_prompt_behavior_stages(self, obs: _model.Observation):
        """Encode normalized per-transition latents and their validity mask."""
        if not self.physical_prompt_behavior_latent_dim:
            raise ValueError("Physical-prompt behavior binding is disabled")
        frame_representations, _, frame_masks, _ = self._encode_physical_prompt_behavior_frames(obs)
        frame_latents = self.physical_prompt_behavior_down(jnp.stack(frame_representations, axis=1))
        return _safe_l2_normalize(frame_latents), jnp.stack(frame_masks, axis=1)

    def encode_physical_prompt_behavior_stage_set(self, obs: _model.Observation):
        """Encode behavior values, visual route keys, and order-sensitive masks."""
        if not self.physical_prompt_visual_stage_routing:
            raise ValueError("Visual stage routing is disabled")
        frame_representations, route_representations, _, order_masks = self._encode_physical_prompt_behavior_frames(obs)
        frame_latents = self.physical_prompt_behavior_down(jnp.stack(frame_representations, axis=1))
        route_keys = self.physical_prompt_stage_route_down(jnp.stack(route_representations, axis=1))
        return (
            _safe_l2_normalize(frame_latents),
            _safe_l2_normalize(route_keys),
            jnp.stack(order_masks, axis=1),
        )

    def encode_physical_prompt_behavior(self, obs: _model.Observation):
        """Encode a normalized behavior latent for the C3 binding objective."""
        if not self.physical_prompt_behavior_latent_dim:
            raise ValueError("Physical-prompt behavior binding is disabled")
        frame_representations, _, frame_masks, _ = self._encode_physical_prompt_behavior_frames(obs)
        return self._pool_behavior_frames(frame_representations, frame_masks)

    def encode_query_stage_key(self, obs: _model.Observation):
        """Encode the live pre-action state without reading query or prompt actions."""
        if not self.physical_prompt_visual_stage_routing:
            raise ValueError("Visual stage routing is disabled")
        live_tokens = self._pool_physical_prompt_image(obs, "base_0_rgb")
        route_representation = jnp.mean(live_tokens, axis=1) + self.physical_prompt_query_state_proj(obs.state)
        return _safe_l2_normalize(self.physical_prompt_stage_route_down(route_representation))

    def encode_query_action_behavior(self, actions: _model.Actions, obs: _model.Observation | None = None):
        """Encode the supervised query action chunk in the same behavior space."""
        if not self.physical_prompt_behavior_latent_dim:
            raise ValueError("Physical-prompt behavior binding is disabled")
        action_mask = jnp.ones(actions.shape[:2], dtype=bool)
        action_token = self._encode_action_sequence(
            actions,
            action_mask,
            self.physical_prompt_query_action_position_embedding.value,
        )
        query_representation = action_token
        if self.physical_prompt_query_context_binding:
            if obs is None:
                raise ValueError("Query-context behavior binding requires an observation")
            state_token = self.physical_prompt_query_state_proj(obs.state)
            live_tokens = self._pool_physical_prompt_image(obs, "base_0_rgb")
            attention_query = _safe_l2_normalize(action_token + state_token)
            attention_keys = _safe_l2_normalize(live_tokens)
            attention_logits = jnp.einsum("bd,bpd->bp", attention_query, attention_keys) / 0.1
            attention = jax.nn.softmax(attention_logits, axis=-1)
            visual_context = jnp.sum(live_tokens * attention[..., None], axis=1)
            live_valid = obs.image_masks["base_0_rgb"].astype(visual_context.dtype)
            query_representation = action_token + state_token + visual_context * live_valid[:, None]
        latent = self.physical_prompt_behavior_down(query_representation)
        return _safe_l2_normalize(latent)

    def score_query_prompt_behavior(
        self,
        query_latent,
        stage_latents,
        stage_mask,
        *,
        temperature: float,
        query_stage_key=None,
        stage_keys=None,
    ):
        """Score a local query against a full demonstration using soft stage alignment."""
        if self.physical_prompt_visual_stage_routing:
            if query_stage_key is None or stage_keys is None:
                raise ValueError("Visual stage routing requires query and demonstration route keys")
            score, _ = _masked_routed_similarity(
                query_latent,
                stage_latents,
                stage_mask,
                query_stage_key,
                stage_keys,
                temperature=temperature,
            )
            return score
        return _masked_logmeanexp_similarity(
            query_latent,
            stage_latents,
            stage_mask,
            temperature=temperature,
        )

    def physical_prompt_stage_routing_weights(self, query_stage_key, stage_keys, stage_mask, *, temperature: float):
        """Expose action-independent route weights for diagnostics and invariance tests."""
        dummy_query = jnp.zeros_like(query_stage_key)
        dummy_candidates = jnp.zeros_like(stage_keys)
        _, weights = _masked_routed_similarity(
            dummy_query,
            dummy_candidates,
            stage_mask,
            query_stage_key,
            stage_keys,
            temperature=temperature,
        )
        return weights

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []

        prompt_names = sorted(
            name
            for name in obs.images
            if name.startswith("physical_prompt_") and not name.startswith("physical_prompt_post_")
        )
        prompt_post_names = sorted(name for name in obs.images if name.startswith("physical_prompt_post_"))
        if self.physical_prompt_frames:
            if len(prompt_names) != self.physical_prompt_frames:
                raise ValueError(
                    f"Expected {self.physical_prompt_frames} physical-prompt frames, got {len(prompt_names)}"
                )
            if obs.physical_prompt_actions is None or obs.physical_prompt_action_mask is None:
                raise ValueError("Physical-prompt actions and mask are required when physical prompting is enabled")
            if self.physical_prompt_effects and len(prompt_post_names) != self.physical_prompt_frames:
                raise ValueError(
                    f"Expected {self.physical_prompt_frames} physical-prompt post frames, got {len(prompt_post_names)}"
                )

            prompt_image_mask = jnp.stack([obs.image_masks[name] for name in prompt_names], axis=1)
            prompt_valid = jnp.any(prompt_image_mask, axis=1)
            boundary_embedding = self.physical_prompt_boundary_embedding.value

            # Explicit boundary tokens and learned frame positions distinguish
            # a demonstration from the live observation while keeping the
            # released PaliGemma backbone intact.
            tokens.append(
                jnp.broadcast_to(
                    boundary_embedding[None, :1],
                    (prompt_valid.shape[0], 1, boundary_embedding.shape[-1]),
                )
            )
            input_mask.append(prompt_valid[:, None])
            ar_mask += [False]

            behavior_frames = []
            behavior_masks = []

            for frame, name in enumerate(prompt_names):
                pre_tokens = self._pool_physical_prompt_image(obs, name)
                position = self.physical_prompt_position_embedding.value[frame]
                image_tokens = pre_tokens + position[None, None, :]
                tokens.append(image_tokens)
                input_mask.append(
                    einops.repeat(
                        obs.image_masks[name],
                        "b -> b s",
                        s=image_tokens.shape[1],
                    )
                )
                ar_mask += [False] * image_tokens.shape[1]

                action_token, action_valid = self._encode_physical_prompt_action(
                    obs.physical_prompt_actions[:, frame], obs.physical_prompt_action_mask[:, frame]
                )
                effect_mask = jnp.logical_and(action_valid, obs.image_masks[name])
                local_effect_tokens = None
                if self.physical_prompt_effects:
                    post_name = prompt_post_names[frame]
                    post_tokens = self._pool_physical_prompt_image(obs, post_name)
                    action_gate = jax.nn.sigmoid(self.physical_prompt_effect_gate(action_token))
                    effect_mask = jnp.logical_and(effect_mask, obs.image_masks[post_name])
                    patch_effects = post_tokens - pre_tokens
                    if self.physical_prompt_local_effect_tokens:
                        # Retain the strongest spatial changes instead of
                        # averaging away contacts and object motion. SigLIP is
                        # frozen, so deterministic top-k routing does not hide
                        # a trainable selection shortcut.
                        _, effect_indices = jax.lax.top_k(
                            jnp.linalg.norm(patch_effects, axis=-1),
                            self.physical_prompt_local_effect_tokens,
                        )
                        selected_effects = jnp.take_along_axis(
                            patch_effects,
                            effect_indices[..., None],
                            axis=1,
                        )
                        effect_latent = jax.nn.gelu(self.physical_prompt_effect_down(selected_effects))
                        local_effect_tokens = self.physical_prompt_effect_up(effect_latent * action_gate[:, None, :])
                        local_effect_tokens += self.physical_prompt_effect_token_embedding.value[None, :, :]
                    else:
                        # C1 compatibility path: fuse one global mean effect
                        # directly into the aligned action token.
                        effect_delta = jnp.mean(patch_effects, axis=1)
                        effect_latent = jax.nn.gelu(self.physical_prompt_effect_down(effect_delta))
                        action_token = action_token + self.physical_prompt_effect_up(effect_latent * action_gate)
                    if self.physical_prompt_behavior_latent_dim:
                        behavior_frames.append(
                            self._behavior_frame_representation(pre_tokens, post_tokens, action_token)
                        )
                        behavior_masks.append(effect_mask)
                action_token = action_token[:, None, :] + position[None, None, :]
                tokens.append(action_token)
                input_mask.append(effect_mask[:, None])
                ar_mask += [False]
                if local_effect_tokens is not None:
                    local_effect_tokens += position[None, None, :]
                    tokens.append(local_effect_tokens)
                    input_mask.append(
                        einops.repeat(
                            effect_mask,
                            "b -> b s",
                            s=self.physical_prompt_local_effect_tokens,
                        )
                    )
                    ar_mask += [False] * self.physical_prompt_local_effect_tokens

            if self.physical_prompt_behavior_latent_dim:
                behavior_latent = self._pool_behavior_frames(behavior_frames, behavior_masks)
                behavior_token = self.physical_prompt_behavior_up(behavior_latent)
                behavior_token += self.physical_prompt_behavior_token_embedding.value[None, :]
                tokens.append(behavior_token[:, None, :])
                input_mask.append(prompt_valid[:, None])
                ar_mask += [False]

            tokens.append(
                jnp.broadcast_to(
                    boundary_embedding[None, 1:],
                    (prompt_valid.shape[0], 1, boundary_embedding.shape[-1]),
                )
            )
            input_mask.append(prompt_valid[:, None])
            ar_mask += [False]

        # Embed the live camera observations after the demonstration.
        for name in obs.images:
            if name.startswith("physical_prompt_"):
                continue
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
