import numpy as np

from openpi.models import model as _model
from openpi.policies import robocasa_policy


def test_inputs_convert_images_and_preserve_prompt() -> None:
    example = robocasa_policy.make_robocasa_example()
    example["observation/base_image"] = np.moveaxis(example["observation/base_image"], -1, 0) / 255.0

    result = robocasa_policy.RoboCasaInputs(_model.ModelType.PI0)(example)

    assert result["image"]["base_0_rgb"].shape == (224, 224, 3)
    assert result["image"]["base_0_rgb"].dtype == np.uint8
    assert result["state"].shape == (16,)
    assert result["prompt"] == "arrange bread in the basket"
    assert all(result["image_mask"].values())


def test_outputs_select_robocasa_action_dimension() -> None:
    actions = np.arange(320).reshape(10, 32)

    result = robocasa_policy.RoboCasaOutputs()({"actions": actions})

    np.testing.assert_array_equal(result["actions"], actions[:, :12])

