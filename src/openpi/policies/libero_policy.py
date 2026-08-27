import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_libero_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType
    physical_prompt_frames: int = 0
    physical_prompt_effects: bool = False
    physical_prompt_counterfactuals: bool = False

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        images = {}
        image_masks = {}
        if self.physical_prompt_frames:
            raw_prompt_images = data.get("physical_prompt/images")
            raw_prompt_actions = data.get("physical_prompt/actions")
            raw_prompt_mask = data.get("physical_prompt/mask")
            raw_prompt_post_images = data.get("physical_prompt/post_images")

            if raw_prompt_images is None:
                prompt_images = np.zeros((self.physical_prompt_frames, *base_image.shape), dtype=base_image.dtype)
                prompt_mask = np.zeros(self.physical_prompt_frames, dtype=bool)
            else:
                if len(raw_prompt_images) != self.physical_prompt_frames:
                    raise ValueError(
                        f"Expected {self.physical_prompt_frames} physical-prompt images, "
                        f"got {len(raw_prompt_images)}"
                    )
                prompt_images = np.stack([_parse_image(image) for image in raw_prompt_images])
                prompt_mask = (
                    np.ones(self.physical_prompt_frames, dtype=bool)
                    if raw_prompt_mask is None
                    else np.asarray(raw_prompt_mask, dtype=bool)
                )

            for frame, image in enumerate(prompt_images):
                key = f"physical_prompt_{frame:03d}_rgb"
                images[key] = image
                image_masks[key] = prompt_mask[frame]

            if self.physical_prompt_effects:
                if raw_prompt_post_images is None:
                    prompt_post_images = np.zeros_like(prompt_images)
                    prompt_post_mask = np.zeros(self.physical_prompt_frames, dtype=bool)
                else:
                    if len(raw_prompt_post_images) != self.physical_prompt_frames:
                        raise ValueError(
                            f"Expected {self.physical_prompt_frames} physical-prompt post images, "
                            f"got {len(raw_prompt_post_images)}"
                        )
                    prompt_post_images = np.stack([_parse_image(image) for image in raw_prompt_post_images])
                    prompt_post_mask = prompt_mask
                for frame, image in enumerate(prompt_post_images):
                    key = f"physical_prompt_post_{frame:03d}_rgb"
                    images[key] = image
                    image_masks[key] = prompt_post_mask[frame]

            if raw_prompt_actions is None:
                prompt_actions = np.zeros((self.physical_prompt_frames, 7), dtype=np.float32)
                prompt_action_mask = np.zeros(self.physical_prompt_frames, dtype=bool)
            else:
                prompt_actions = np.asarray(raw_prompt_actions, dtype=np.float32)
                if prompt_actions.shape[0] != self.physical_prompt_frames:
                    raise ValueError(
                        f"Expected {self.physical_prompt_frames} physical-prompt actions, "
                        f"got {prompt_actions.shape[0]}"
                    )
                prompt_action_mask = prompt_mask.copy()

            if self.physical_prompt_counterfactuals:
                raw_counterfactual_images = data.get("physical_prompt/counterfactual_images")
                raw_counterfactual_post_images = data.get("physical_prompt/counterfactual_post_images")
                raw_counterfactual_actions = data.get("physical_prompt/counterfactual_actions")
                raw_counterfactual_mask = data.get("physical_prompt/counterfactual_mask")
                if raw_counterfactual_images is None:
                    counterfactual_images = np.zeros_like(prompt_images)
                    counterfactual_mask = np.zeros(self.physical_prompt_frames, dtype=bool)
                else:
                    if len(raw_counterfactual_images) != self.physical_prompt_frames:
                        raise ValueError(
                            f"Expected {self.physical_prompt_frames} counterfactual-prompt images, "
                            f"got {len(raw_counterfactual_images)}"
                        )
                    counterfactual_images = np.stack([_parse_image(image) for image in raw_counterfactual_images])
                    counterfactual_mask = (
                        np.ones(self.physical_prompt_frames, dtype=bool)
                        if raw_counterfactual_mask is None
                        else np.asarray(raw_counterfactual_mask, dtype=bool)
                    )
                counterfactual_image_dict = {
                    f"physical_prompt_{frame:03d}_rgb": image for frame, image in enumerate(counterfactual_images)
                }
                counterfactual_image_mask_dict = {
                    f"physical_prompt_{frame:03d}_rgb": counterfactual_mask[frame]
                    for frame in range(self.physical_prompt_frames)
                }
                if self.physical_prompt_effects:
                    if raw_counterfactual_post_images is None:
                        counterfactual_post_images = np.zeros_like(counterfactual_images)
                        counterfactual_post_mask = np.zeros(self.physical_prompt_frames, dtype=bool)
                    else:
                        if len(raw_counterfactual_post_images) != self.physical_prompt_frames:
                            raise ValueError(
                                f"Expected {self.physical_prompt_frames} counterfactual-prompt post images, "
                                f"got {len(raw_counterfactual_post_images)}"
                            )
                        counterfactual_post_images = np.stack(
                            [_parse_image(image) for image in raw_counterfactual_post_images]
                        )
                        counterfactual_post_mask = counterfactual_mask
                    counterfactual_image_dict.update(
                        {
                            f"physical_prompt_post_{frame:03d}_rgb": image
                            for frame, image in enumerate(counterfactual_post_images)
                        }
                    )
                    counterfactual_image_mask_dict.update(
                        {
                            f"physical_prompt_post_{frame:03d}_rgb": counterfactual_post_mask[frame]
                            for frame in range(self.physical_prompt_frames)
                        }
                    )
                if raw_counterfactual_actions is None:
                    counterfactual_actions = np.zeros((self.physical_prompt_frames, 7), dtype=np.float32)
                    counterfactual_action_mask = np.zeros(self.physical_prompt_frames, dtype=bool)
                else:
                    counterfactual_actions = np.asarray(raw_counterfactual_actions, dtype=np.float32)
                    if counterfactual_actions.shape[0] != self.physical_prompt_frames:
                        raise ValueError(
                            f"Expected {self.physical_prompt_frames} counterfactual-prompt actions, "
                            f"got {counterfactual_actions.shape[0]}"
                        )
                    counterfactual_action_mask = counterfactual_mask.copy()

        # Create inputs dict. Prompt frames deliberately precede the live
        # cameras in insertion order; the model also checks their key prefix.
        images.update(
            {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            }
        )
        image_masks.update(
            {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            }
        )
        inputs = {
            "state": data["observation/state"],
            "image": images,
            "image_mask": image_masks,
        }
        if self.physical_prompt_frames:
            inputs["physical_prompt_actions"] = prompt_actions
            inputs["physical_prompt_action_mask"] = prompt_action_mask
            if self.physical_prompt_counterfactuals:
                inputs["physical_prompt_counterfactual_images"] = counterfactual_image_dict
                inputs["physical_prompt_counterfactual_image_masks"] = counterfactual_image_mask_dict
                inputs["physical_prompt_counterfactual_actions"] = counterfactual_actions
                inputs["physical_prompt_counterfactual_action_mask"] = counterfactual_action_mask
                inputs["physical_prompt_rank_mask"] = np.asarray(
                    data.get("physical_prompt/rank_mask", False), dtype=bool
                )

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        return {"actions": np.asarray(data["actions"][:, :7])}
