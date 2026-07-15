import dataclasses

import numpy as np

from openpi import transforms
from openpi.policies import aloha_policy


AV_ALOHA_ACTION_DIM = 21
TASK_ARM_ACTION_DIM = 14


@dataclasses.dataclass(frozen=True)
class AvAlohaMotorInputs(transforms.DataTransformFn):
    """Map AV-ALOHA demonstrations to a fixed-camera, motor-only policy.

    The source state and action contain two seven-dimensional task arms followed
    by a seven-dimensional active-camera arm.  Candidate-level evidence
    experiments intentionally train only the first 14 dimensions so camera
    queries cannot silently regenerate the motor candidate support.
    """

    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"])
        if state.shape != (AV_ALOHA_ACTION_DIM,):
            raise ValueError(
                f"AV-ALOHA state must have shape ({AV_ALOHA_ACTION_DIM},), "
                f"got {state.shape}"
            )

        images = data["images"]
        expected_images = {"overhead_cam", "wrist_cam_left", "wrist_cam_right"}
        if set(images) != expected_images:
            raise ValueError(
                f"AV-ALOHA motor policy expects images {sorted(expected_images)}, "
                f"got {sorted(images)}"
            )

        aloha_data = {
            "images": {
                "cam_high": images["overhead_cam"],
                "cam_left_wrist": images["wrist_cam_left"],
                "cam_right_wrist": images["wrist_cam_right"],
            },
            "state": state[:TASK_ARM_ACTION_DIM].copy(),
        }
        if "actions" in data:
            actions = np.asarray(data["actions"])
            if actions.ndim != 2 or actions.shape[-1] != AV_ALOHA_ACTION_DIM:
                raise ValueError(
                    "AV-ALOHA actions must have shape [horizon, 21], "
                    f"got {actions.shape}"
                )
            aloha_data["actions"] = actions[:, :TASK_ARM_ACTION_DIM].copy()
        if "prompt" in data:
            aloha_data["prompt"] = data["prompt"]

        return aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)(aloha_data)


@dataclasses.dataclass(frozen=True)
class AvAlohaMotorOutputs(transforms.DataTransformFn):
    """Decode only the two task arms from a padded OpenPI action chunk."""

    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        return aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)(data)
