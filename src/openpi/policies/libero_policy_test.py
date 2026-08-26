import numpy as np

from openpi.models import model as _model
from openpi.policies import libero_policy


def _example_with_physical_prompt(num_frames: int = 3) -> dict:
    return {
        "observation/state": np.zeros(8, dtype=np.float32),
        "observation/image": np.zeros((3, 24, 32), dtype=np.float32),
        "observation/wrist_image": np.zeros((3, 24, 32), dtype=np.float32),
        "physical_prompt/images": np.zeros((num_frames, 3, 24, 32), dtype=np.float32),
        "physical_prompt/actions": np.arange(num_frames * 7, dtype=np.float32).reshape(num_frames, 7),
        "physical_prompt/mask": np.ones(num_frames, dtype=bool),
        "prompt": "Follow the demonstrated behavior.",
    }


def test_physical_prompt_is_aligned_and_precedes_live_images() -> None:
    transform = libero_policy.LiberoInputs(model_type=_model.ModelType.PI05, physical_prompt_frames=3)
    result = transform(_example_with_physical_prompt())

    assert list(result["image"]) == [
        "physical_prompt_000_rgb",
        "physical_prompt_001_rgb",
        "physical_prompt_002_rgb",
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    ]
    assert result["physical_prompt_actions"].shape == (3, 7)
    assert result["physical_prompt_action_mask"].tolist() == [True, True, True]


def test_missing_physical_prompt_produces_masked_static_shape() -> None:
    transform = libero_policy.LiberoInputs(model_type=_model.ModelType.PI05, physical_prompt_frames=3)
    example = _example_with_physical_prompt()
    del example["physical_prompt/images"]
    del example["physical_prompt/actions"]
    del example["physical_prompt/mask"]

    result = transform(example)

    assert result["physical_prompt_actions"].shape == (3, 7)
    assert not result["physical_prompt_action_mask"].any()
    assert not any(result["image_mask"][f"physical_prompt_{frame:03d}_rgb"] for frame in range(3))


def test_effect_binding_prompt_adds_post_images() -> None:
    transform = libero_policy.LiberoInputs(
        model_type=_model.ModelType.PI05,
        physical_prompt_frames=3,
        physical_prompt_effects=True,
    )
    example = _example_with_physical_prompt()
    example["physical_prompt/post_images"] = np.ones((3, 3, 24, 32), dtype=np.float32)

    result = transform(example)

    assert all(f"physical_prompt_post_{frame:03d}_rgb" in result["image"] for frame in range(3))
    assert all(result["image_mask"][f"physical_prompt_post_{frame:03d}_rgb"] for frame in range(3))


def test_counterfactual_prompt_is_kept_out_of_the_live_image_dict() -> None:
    transform = libero_policy.LiberoInputs(
        model_type=_model.ModelType.PI05,
        physical_prompt_frames=3,
        physical_prompt_effects=True,
        physical_prompt_counterfactuals=True,
    )
    example = _example_with_physical_prompt()
    example.update(
        {
            "physical_prompt/post_images": np.ones((3, 3, 24, 32), dtype=np.float32),
            "physical_prompt/counterfactual_images": np.full((3, 3, 24, 32), 0.5, dtype=np.float32),
            "physical_prompt/counterfactual_post_images": np.full((3, 3, 24, 32), 0.75, dtype=np.float32),
            "physical_prompt/counterfactual_actions": np.ones((3, 7), dtype=np.float32),
            "physical_prompt/counterfactual_mask": np.ones(3, dtype=bool),
        }
    )

    result = transform(example)

    assert "physical_prompt_counterfactual_images" in result
    assert not any("counterfactual" in key for key in result["image"])
    assert result["physical_prompt_counterfactual_actions"].shape == (3, 7)
