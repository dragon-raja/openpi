import numpy as np
import pytest

from openpi.training import config
from openpi.training import data_loader


def _raw_sample() -> dict:
    return {
        "observation.images.overhead_cam": np.zeros((3, 8, 10), dtype=np.float32),
        "observation.images.wrist_cam_left": np.full(
            (3, 8, 10), 0.5, dtype=np.float32
        ),
        "observation.images.wrist_cam_right": np.ones(
            (3, 8, 10), dtype=np.float32
        ),
        "observation.state": np.linspace(-0.5, 0.5, 21, dtype=np.float32),
        "action": np.linspace(-1.0, 1.0, 4 * 21, dtype=np.float32).reshape(
            4, 21
        ),
        "prompt": "thread the needle",
    }


def test_motor_config_preserves_prompt_and_roundtrips_task_actions(tmp_path) -> None:
    train_config = config.get_config("pi05_av_aloha_sewneedle_motor_v0")
    assert train_config.weight_loader.params_path == (
        "/workspace/ckpt_download/openpi-assets/checkpoints/pi05_base/params"
    )
    assert train_config.batch_size == 32
    assert train_config.num_workers == 16
    assert train_config.fsdp_devices == 4
    data_config = train_config.data.create(tmp_path, train_config.model)
    raw = _raw_sample()

    transformed = raw
    for transform in data_config.repack_transforms.inputs:
        transformed = transform(transformed)
    for transform in data_config.data_transforms.inputs:
        transformed = transform(transformed)

    assert transformed["state"].shape == (14,)
    assert transformed["actions"].shape == (4, 14)
    assert transformed["prompt"] == "thread the needle"
    assert set(transformed["image"]) == {
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    }

    decoded = {
        "state": transformed["state"],
        "actions": transformed["actions"].copy(),
    }
    for transform in data_config.data_transforms.outputs:
        decoded = transform(decoded)
    np.testing.assert_allclose(decoded["actions"], raw["action"][:, :14])


def test_av_aloha_uses_dedicated_local_root_and_video_backend(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENPI_AV_ALOHA_LEROBOT_ROOT", "/tmp/av-aloha-v20")
    monkeypatch.delenv("OPENPI_AV_ALOHA_VIDEO_BACKEND", raising=False)
    repo_id = "iantc104/gv_sim_sew_needle_3arms"

    assert data_loader._lerobot_root_for_repo(repo_id) == "/tmp/av-aloha-v20"
    assert data_loader._lerobot_video_backend_for_repo(repo_id) == "pyav"


def test_av_aloha_refuses_implicit_hub_download(monkeypatch) -> None:
    monkeypatch.delenv("OPENPI_AV_ALOHA_LEROBOT_ROOT", raising=False)

    with pytest.raises(
        RuntimeError, match="refusing an implicit Hugging Face download"
    ):
        data_loader._lerobot_root_for_repo("iantc104/gv_sim_sew_needle_3arms")
