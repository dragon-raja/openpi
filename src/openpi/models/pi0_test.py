import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

import openpi.models.pi0 as _pi0
import openpi.models.pi0_config as _pi0_config


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


def test_local_effect_tokens_require_effect_prompts():
    with pytest.raises(ValueError, match="requires physical_prompt_effects"):
        _pi0_config.Pi0Config(physical_prompt_frames=8, physical_prompt_local_effect_tokens=4)

    config = _pi0_config.Pi0Config(
        physical_prompt_frames=8,
        physical_prompt_pool_grid=4,
        physical_prompt_effects=True,
        physical_prompt_local_effect_tokens=4,
    )
    assert config.physical_prompt_local_effect_tokens == 4


def test_behavior_binding_requires_exact_multistep_effects():
    with pytest.raises(ValueError, match="requires multi-step"):
        _pi0_config.Pi0Config(
            physical_prompt_frames=8,
            physical_prompt_effects=True,
            physical_prompt_behavior_latent_dim=128,
        )

    config = _pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        physical_prompt_frames=8,
        physical_prompt_effects=True,
        physical_prompt_effect_horizons=(1, 2, 4, 8),
        physical_prompt_behavior_latent_dim=128,
        physical_prompt_counterfactuals=True,
    )
    observation, _ = config.inputs_spec(batch_size=2)

    assert observation.physical_prompt_actions.shape == (2, 8, 8, config.action_dim)
    assert observation.physical_prompt_action_mask.shape == (2, 8, 8)
    assert observation.physical_prompt_counterfactual_actions.shape == (2, 8, 8, config.action_dim)

    with pytest.raises(ValueError, match="requires a behavior latent"):
        _pi0_config.Pi0Config(
            physical_prompt_frames=8,
            physical_prompt_effects=True,
            physical_prompt_effect_horizons=(1, 2),
            physical_prompt_directed_action_flow=True,
        )

    with pytest.raises(ValueError, match="requires a behavior latent"):
        _pi0_config.Pi0Config(physical_prompt_query_context_binding=True)

    with pytest.raises(ValueError, match="requires a behavior latent"):
        _pi0_config.Pi0Config(physical_prompt_stage_alignment=True)

    with pytest.raises(ValueError, match="requires stage alignment"):
        _pi0_config.Pi0Config(
            physical_prompt_frames=8,
            physical_prompt_effects=True,
            physical_prompt_effect_horizons=(1, 2),
            physical_prompt_behavior_latent_dim=128,
            physical_prompt_visual_stage_routing=True,
        )


def test_safe_l2_normalize_has_finite_zero_vector_gradient():
    value = jnp.zeros((2, 256), dtype=jnp.float32)
    normalized = _pi0._safe_l2_normalize(value)  # noqa: SLF001
    gradient = jax.grad(lambda x: jnp.sum(_pi0._safe_l2_normalize(x)))(value)  # noqa: SLF001

    assert jnp.array_equal(normalized, value)
    assert jnp.all(jnp.isfinite(gradient))


def test_masked_stage_alignment_ignores_padding_and_handles_empty_sets():
    query = _pi0._safe_l2_normalize(jnp.array([[1.0, 0.0], [1.0, 0.0]]))  # noqa: SLF001
    candidates = _pi0._safe_l2_normalize(  # noqa: SLF001
        jnp.array([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]])
    )
    mask = jnp.array([[True, True, False], [False, False, False]])

    score = _pi0._masked_logmeanexp_similarity(query, candidates, mask, temperature=0.1)  # noqa: SLF001
    expected = 0.1 * (jax.nn.logsumexp(jnp.array([10.0, 0.0])) - jnp.log(2.0))

    assert score[0] == pytest.approx(float(expected), abs=2e-6)
    assert score[1] == 0.0
    assert jnp.all(
        jnp.isfinite(
            jax.grad(
                lambda q: jnp.sum(
                    _pi0._masked_logmeanexp_similarity(  # noqa: SLF001
                        q, candidates, mask, temperature=0.1
                    )
                )
            )(query)
        )
    )


def test_visual_routing_is_independent_of_behavior_candidate_values():
    route_query = _pi0._safe_l2_normalize(jnp.array([[1.0, 0.0]]))  # noqa: SLF001
    route_keys = _pi0._safe_l2_normalize(jnp.array([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]]))  # noqa: SLF001
    query = _pi0._safe_l2_normalize(jnp.array([[1.0, 1.0]]))  # noqa: SLF001
    forward = _pi0._safe_l2_normalize(jnp.array([[[1.0, 1.0], [1.0, -1.0], [-1.0, 0.0]]]))  # noqa: SLF001
    reversed_values = -forward
    mask = jnp.array([[True, True, False]])

    forward_score, forward_weights = _pi0._masked_routed_similarity(  # noqa: SLF001
        query, forward, mask, route_query, route_keys, temperature=0.1
    )
    reversed_score, reversed_weights = _pi0._masked_routed_similarity(  # noqa: SLF001
        query, reversed_values, mask, route_query, route_keys, temperature=0.1
    )

    assert jnp.array_equal(forward_weights, reversed_weights)
    assert forward_weights[0, 2] == 0.0
    assert forward_score[0] == pytest.approx(-float(reversed_score[0]))


def test_masked_action_flow_changes_sign_under_valid_prefix_reversal():
    tokens = jnp.array([[[0.0], [1.0], [3.0], [99.0]], [[4.0], [9.0], [99.0], [99.0]]])
    mask = jnp.array([[True, True, True, False], [True, False, False, False]])
    reversed_tokens = jnp.array([[[3.0], [1.0], [0.0], [99.0]], [[4.0], [9.0], [99.0], [99.0]]])

    forward = _pi0._masked_action_flow(tokens, mask)  # noqa: SLF001
    backward = _pi0._masked_action_flow(reversed_tokens, mask)  # noqa: SLF001

    assert jnp.allclose(forward[0], -backward[0])
    assert jnp.array_equal(forward[1], jnp.zeros(1))
    assert jnp.array_equal(backward[1], jnp.zeros(1))
