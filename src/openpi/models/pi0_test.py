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


def test_safe_l2_normalize_has_finite_zero_vector_gradient():
    value = jnp.zeros((2, 256), dtype=jnp.float32)
    normalized = _pi0._safe_l2_normalize(value)  # noqa: SLF001
    gradient = jax.grad(lambda x: jnp.sum(_pi0._safe_l2_normalize(x)))(value)  # noqa: SLF001

    assert jnp.array_equal(normalized, value)
    assert jnp.all(jnp.isfinite(gradient))
