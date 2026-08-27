import dataclasses

import jax
import torch

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


class _PromptDatasetFixture:
    def __init__(self):
        self.meta = type(
            "Meta",
            (),
            {
                "tasks": {0: "move the object"},
                "episodes": {
                    0: {"tasks": ["move the object"], "length": 4},
                    1: {"tasks": ["move the object"], "length": 4},
                },
            },
        )()
        self.episode_data_index = {
            "from": torch.tensor([0, 4]),
            "to": torch.tensor([4, 8]),
        }

    def __len__(self):
        return 8

    def __getitem__(self, index):
        episode = index // 4
        return {
            "image": torch.full((3, 2, 2), index, dtype=torch.float32),
            "actions": torch.full((2, 7), index, dtype=torch.float32),
            "episode_index": torch.tensor(episode),
            "task_index": torch.tensor(0),
        }


class _CounterfactualPromptDatasetFixture:
    def __init__(self):
        self.meta = type(
            "Meta",
            (),
            {
                "tasks": {0: "task zero", 1: "task one"},
                "episodes": {
                    0: {"tasks": ["task zero"], "length": 4},
                    1: {"tasks": ["task zero"], "length": 4},
                    2: {"tasks": ["task one"], "length": 4},
                    3: {"tasks": ["task one"], "length": 4},
                },
            },
        )()
        self.episode_data_index = {
            "from": torch.tensor([0, 4, 8, 12]),
            "to": torch.tensor([4, 8, 12, 16]),
        }

    def __len__(self):
        return 16

    def __getitem__(self, index):
        episode = index // 4
        return {
            "image": torch.full((3, 2, 2), index, dtype=torch.float32),
            "actions": torch.full((2, 7), index, dtype=torch.float32),
            "episode_index": torch.tensor(episode),
            "task_index": torch.tensor(episode // 2),
        }


def test_physical_prompt_dataset_uses_other_episode_and_is_deterministic():
    dataset = _data_loader.PhysicalPromptDataset(_PromptDatasetFixture(), num_frames=3, seed=7)

    first = dataset[0]
    repeated = dataset[0]

    assert int(first["physical_prompt_episode_index"]) == 1
    assert torch.equal(first["physical_prompt_images"], repeated["physical_prompt_images"])
    assert first["physical_prompt_actions"][:, 0].tolist() == [4.0, 5.0, 7.0]
    assert first["physical_prompt_mask"].tolist() == [True, True, True]


def test_physical_prompt_dataset_builds_action_effect_transitions():
    dataset = _data_loader.PhysicalPromptDataset(_PromptDatasetFixture(), num_frames=3, seed=7, include_effects=True)

    item = dataset[0]

    assert item["physical_prompt_images"][:, 0, 0, 0].tolist() == [4.0, 5.0, 6.0]
    assert item["physical_prompt_post_images"][:, 0, 0, 0].tolist() == [5.0, 6.0, 7.0]
    assert item["physical_prompt_actions"][:, 0].tolist() == [4.0, 5.0, 6.0]


def test_physical_prompt_dataset_supplies_an_explicit_wrong_task():
    dataset = _data_loader.PhysicalPromptDataset(
        _CounterfactualPromptDatasetFixture(),
        num_frames=3,
        seed=7,
        include_effects=True,
        include_counterfactuals=True,
    )

    item = dataset[0]

    assert int(item["physical_prompt_counterfactual_task_index"]) == 1
    assert item["physical_prompt_counterfactual_images"].shape == (3, 3, 2, 2)
    assert not torch.equal(item["physical_prompt_images"], item["physical_prompt_counterfactual_images"])
    assert bool(item["physical_prompt_rank_mask"])


def test_physical_prompt_dataset_masks_easy_negatives_and_builds_language_anchors():
    dataset = _data_loader.PhysicalPromptDataset(
        _CounterfactualPromptDatasetFixture(),
        num_frames=3,
        seed=7,
        include_effects=True,
        include_counterfactuals=True,
        hard_negatives_only=True,
        language_anchor_fraction=1.0,
    )

    item = dataset[0]

    assert item["prompt"] == "task zero"
    assert not item["physical_prompt_mask"].any()
    assert int(item["physical_prompt_counterfactual_task_index"]) == 0
    assert not bool(item["physical_prompt_rank_mask"])


def test_episode_block_sampler_preserves_blocks_and_changes_epoch_order():
    boundaries = {
        "from": torch.tensor([0, 5]),
        "to": torch.tensor([5, 10]),
    }
    sampler = _data_loader.EpisodeBlockSampler(boundaries, block_size=3, seed=4)

    epoch_a = list(sampler)
    epoch_b = list(sampler)

    assert sorted(epoch_a) == list(range(10))
    assert sorted(epoch_b) == list(range(10))
    assert epoch_a != epoch_b
    # The sampler never permutes frames within a block.  Short tail blocks can
    # appear at arbitrary positions, so verify the defining adjacency instead.
    assert all(epoch_a.index(value + 1) == epoch_a.index(value) + 1 for value in (0, 1, 3, 5, 6, 8))


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
