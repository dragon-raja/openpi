import dataclasses
import os
import pathlib

import jax.numpy as jnp
import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.models import model as _model
from openpi.training import config as _config

from . import train


def test_physical_prompt_rank_schedule_and_per_example_reduction():
    config = dataclasses.replace(
        _config._CONFIGS_DICT["debug"],  # noqa: SLF001
        physical_prompt_rank_warmup_steps=10,
        physical_prompt_rank_ramp_steps=20,
    )

    assert float(train._scheduled_rank_weight(config, 9, 0.2)) == 0.0  # noqa: SLF001
    assert float(train._scheduled_rank_weight(config, 20, 0.2)) == pytest.approx(0.1)  # noqa: SLF001
    assert float(train._scheduled_rank_weight(config, 30, 0.2)) == pytest.approx(0.2)  # noqa: SLF001
    loss = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4)
    assert jnp.array_equal(train._per_example_loss(loss), jnp.array([5.5, 17.5]))  # noqa: SLF001


def test_physical_prompt_interventions_only_change_the_prompt():
    observation = _model.Observation(
        images={
            "physical_prompt_000_rgb": jnp.arange(2, dtype=jnp.float32)[:, None, None, None],
            "physical_prompt_post_000_rgb": jnp.arange(2, 4, dtype=jnp.float32)[:, None, None, None],
            "base_0_rgb": jnp.arange(4, 6, dtype=jnp.float32)[:, None, None, None],
        },
        image_masks={
            "physical_prompt_000_rgb": jnp.ones(2, dtype=bool),
            "physical_prompt_post_000_rgb": jnp.ones(2, dtype=bool),
            "base_0_rgb": jnp.ones(2, dtype=bool),
        },
        state=jnp.zeros((2, 1)),
        physical_prompt_actions=jnp.arange(12, dtype=jnp.float32).reshape(2, 3, 2),
        physical_prompt_action_mask=jnp.ones((2, 3), dtype=bool),
        physical_prompt_counterfactual_images={
            "physical_prompt_000_rgb": jnp.arange(6, 8, dtype=jnp.float32)[:, None, None, None],
            "physical_prompt_post_000_rgb": jnp.arange(8, 10, dtype=jnp.float32)[:, None, None, None],
        },
        physical_prompt_counterfactual_image_masks={
            "physical_prompt_000_rgb": jnp.ones(2, dtype=bool),
            "physical_prompt_post_000_rgb": jnp.ones(2, dtype=bool),
        },
        physical_prompt_counterfactual_actions=jnp.arange(12, 24, dtype=jnp.float32).reshape(2, 3, 2),
        physical_prompt_counterfactual_action_mask=jnp.ones((2, 3), dtype=bool),
    )

    counterfactual = train._intervene_physical_prompt(observation, use_counterfactual=True)  # noqa: SLF001
    reversed_actions = train._intervene_physical_prompt(observation, reverse_actions=True)  # noqa: SLF001

    assert jnp.array_equal(counterfactual.images["base_0_rgb"], observation.images["base_0_rgb"])
    assert jnp.array_equal(
        counterfactual.images["physical_prompt_000_rgb"],
        observation.physical_prompt_counterfactual_images["physical_prompt_000_rgb"],
    )
    assert jnp.array_equal(
        reversed_actions.physical_prompt_actions,
        observation.physical_prompt_actions[:, ::-1],
    )


@pytest.mark.parametrize("config_name", ["debug"])
def test_train(tmp_path: pathlib.Path, config_name: str):
    config = dataclasses.replace(
        _config._CONFIGS_DICT[config_name],  # noqa: SLF001
        batch_size=2,
        checkpoint_base_dir=str(tmp_path / "checkpoint"),
        exp_name="test",
        overwrite=False,
        resume=False,
        num_train_steps=2,
        log_interval=1,
    )
    train.main(config)

    # test resuming
    config = dataclasses.replace(config, resume=True, num_train_steps=4)
    train.main(config)
