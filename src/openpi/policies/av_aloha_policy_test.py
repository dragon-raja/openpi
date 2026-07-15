import numpy as np
import pytest

from openpi.policies import av_aloha_policy


def _example() -> dict:
    return {
        "images": {
            "overhead_cam": np.zeros((3, 8, 10), dtype=np.float32),
            "wrist_cam_left": np.full((3, 8, 10), 0.5, dtype=np.float32),
            "wrist_cam_right": np.ones((3, 8, 10), dtype=np.float32),
        },
        "state": np.arange(21, dtype=np.float32),
        "actions": np.arange(4 * 21, dtype=np.float32).reshape(4, 21),
        "prompt": "thread the needle",
    }


def test_motor_inputs_drop_camera_arm_without_changing_task_arms() -> None:
    example = _example()
    result = av_aloha_policy.AvAlohaMotorInputs(adapt_to_pi=False)(example)

    np.testing.assert_array_equal(result["state"], example["state"][:14])
    np.testing.assert_array_equal(result["actions"], example["actions"][:, :14])
    assert result["prompt"] == "thread the needle"
    assert result["image"]["base_0_rgb"].shape == (8, 10, 3)
    assert result["image"]["base_0_rgb"].dtype == np.uint8
    assert set(result["image"]) == {
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    }


def test_motor_inputs_reject_wrong_state_or_action_contract() -> None:
    example = _example()
    example["state"] = example["state"][:14]
    with pytest.raises(ValueError, match="state must have shape"):
        av_aloha_policy.AvAlohaMotorInputs()(example)

    example = _example()
    example["actions"] = example["actions"][:, :14]
    with pytest.raises(ValueError, match="actions must have shape"):
        av_aloha_policy.AvAlohaMotorInputs()(example)


def test_motor_outputs_return_only_task_arm_dimensions() -> None:
    actions = np.arange(5 * 32, dtype=np.float32).reshape(5, 32)

    result = av_aloha_policy.AvAlohaMotorOutputs(adapt_to_pi=False)(
        {"actions": actions}
    )

    np.testing.assert_array_equal(result["actions"], actions[:, :14])
