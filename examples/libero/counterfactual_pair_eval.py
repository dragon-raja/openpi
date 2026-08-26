"""Counterfactual same-scene evaluation for LIBERO policies.

Each task pair has identical physics and initial conditions but two alternative
goal predicates. Both environments receive the same initial simulator state and
the same actions, so the first goal reached identifies the policy's behavior.
This prevents scene layout from revealing which member of the pair is expected.
"""

import argparse
import collections
import dataclasses
import hashlib
import json
import math
import pathlib
import time

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
POLICY_NOISE_KEY = "_openpi_noise"
NEUTRAL_PROMPT = "Follow the demonstrated behavior."
STATE_ATOL = 1e-12


@dataclasses.dataclass(frozen=True)
class PairSpec:
    pair_id: str
    suite: str
    task_a: str
    task_b: str


@dataclasses.dataclass(frozen=True)
class PhysicalPrompt:
    task: str
    demo_episode_index: int
    images: np.ndarray
    actions: np.ndarray
    mask: np.ndarray
    post_images: np.ndarray | None = None

    def metadata(self, *, action_order: str) -> dict:
        metadata = {
            "task": self.task,
            "demo_episode_index": self.demo_episode_index,
            "num_frames": int(len(self.images)),
            "image_sha256": _sha256(self.images),
            "action_sha256": _sha256(self.actions),
            "action_order": action_order,
        }
        if self.post_images is not None:
            metadata["post_image_sha256"] = _sha256(self.post_images)
        return metadata


@dataclasses.dataclass(frozen=True)
class PromptCondition:
    name: str
    language: str
    expected_goal: str | None
    physical_prompt: PhysicalPrompt | None = None
    action_order: str = "aligned"


PAIR_SPECS = {
    spec.pair_id: spec
    for spec in (
        PairSpec(
            "red_mug_plate_left_right",
            "libero_90",
            "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate",
            "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_right_plate",
        ),
        PairSpec(
            "pudding_plate_left_right",
            "libero_90",
            "LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_left_of_the_plate",
            "LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_right_of_the_plate",
        ),
        PairSpec(
            "black_bowl_front_back_binding",
            "libero_90",
            "KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate",
            "KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate",
        ),
        PairSpec(
            "black_bowl_plate_cabinet",
            "libero_90",
            "KITCHEN_SCENE1_put_the_black_bowl_on_the_plate",
            "KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet",
        ),
        PairSpec(
            "wine_bottle_drawer_rack",
            "libero_90",
            "KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet",
            "KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack",
        ),
        PairSpec(
            "black_bowl_drawer_plate",
            "libero_90",
            "KITCHEN_SCENE5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet",
            "KITCHEN_SCENE5_put_the_black_bowl_on_the_plate",
        ),
        PairSpec(
            "frying_pan_shelf_under",
            "libero_90",
            "KITCHEN_SCENE9_put_the_frying_pan_on_the_cabinet_shelf",
            "KITCHEN_SCENE9_put_the_frying_pan_under_the_cabinet_shelf",
        ),
        PairSpec(
            "study_book_shelf_under",
            "libero_90",
            "STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_on_the_cabinet_shelf",
            "STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_under_the_cabinet_shelf",
        ),
        PairSpec(
            "caddy_scene1_left_right",
            "libero_90",
            "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy",
            "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy",
        ),
        PairSpec(
            "caddy_scene2_left_right",
            "libero_90",
            "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy",
            "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy",
        ),
        PairSpec(
            "caddy_scene3_left_right",
            "libero_90",
            "STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy",
            "STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy",
        ),
        PairSpec(
            "id_libero10_scene2_object_sets",
            "libero_10",
            "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
            "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
        ),
        PairSpec(
            "id_goal_bowl_plate_stove",
            "libero_goal",
            "put_the_bowl_on_the_plate",
            "put_the_bowl_on_the_stove",
        ),
        PairSpec(
            "id_goal_wine_rack_cabinet",
            "libero_goal",
            "put_the_wine_bottle_on_the_rack",
            "put_the_wine_bottle_on_top_of_the_cabinet",
        ),
    )
}


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_int_csv(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in _parse_csv(value))


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _remove_bddl_sections(text: str, names: tuple[str, ...]) -> str:
    """Remove complete top-level BDDL sections, including nested forms."""
    lines = text.splitlines()
    kept: list[str] = []
    skipping = False
    depth = 0
    for line in lines:
        stripped = line.strip()
        if not skipping and any(stripped.startswith(f"(:{name}") for name in names):
            skipping = True
            depth = line.count("(") - line.count(")")
            if depth <= 0:
                skipping = False
            continue
        if skipping:
            depth += line.count("(") - line.count(")")
            if depth <= 0:
                skipping = False
            continue
        kept.append(line.rstrip())
    return "\n".join(kept).strip() + "\n"


def _task_lookup(task_suite) -> dict[str, tuple[int, object]]:
    return {task.name: (index, task) for index, task in enumerate(task_suite.tasks)}


def _task_bddl_path(task) -> pathlib.Path:
    return pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file


def _make_env(task, seed: int) -> OffScreenRenderEnv:
    # LIBERO's default hard reset rebuilds MJCF and can change model-level
    # nuisance variables that are absent from a flattened MuJoCo state. Freeze
    # both construction randomness and MJCF across paired interventions.
    np.random.seed(seed)
    env = OffScreenRenderEnv(
        bddl_file_name=_task_bddl_path(task),
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
        hard_reset=False,
    )
    env.seed(seed)
    # The constructor performs one seeded placement. Preserve its fixture body
    # poses on later resets; flattened MuJoCo states only restore qpos / qvel.
    env.env.deterministic_reset = True
    return env


def _max_state_difference(env_a, env_b) -> float:
    state_a = np.asarray(env_a.env.sim.get_state().flatten())
    state_b = np.asarray(env_b.env.sim.get_state().flatten())
    if state_a.shape != state_b.shape:
        return math.inf
    return float(np.max(np.abs(state_a - state_b)))


def _observation_differences(obs_a: dict, obs_b: dict) -> dict[str, float]:
    keys = (
        "agentview_image",
        "robot0_eye_in_hand_image",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    )
    differences = {}
    for key in keys:
        left = np.asarray(obs_a[key])
        right = np.asarray(obs_b[key])
        differences[key] = math.inf if left.shape != right.shape else float(np.max(np.abs(left - right)))
    return differences


def _copy_observation(obs: dict) -> dict[str, np.ndarray]:
    return {key: np.array(value, copy=True) for key, value in obs.items() if isinstance(value, np.ndarray)}


def _reset_pair(env_a, env_b, initial_state: np.ndarray, wait_steps: int):
    env_a.reset()
    env_b.reset()
    obs_a = env_a.set_init_state(initial_state)
    obs_b = env_b.set_init_state(initial_state)
    initial_differences = _observation_differences(obs_a, obs_b)
    initial_state_difference = _max_state_difference(env_a, env_b)
    for _ in range(wait_steps):
        obs_a, _, done_a, _ = env_a.step(LIBERO_DUMMY_ACTION)
        obs_b, _, done_b, _ = env_b.step(LIBERO_DUMMY_ACTION)
        if done_a or done_b:
            raise RuntimeError("A counterfactual goal is already satisfied during stabilization")
    return obs_a, obs_b, initial_differences, initial_state_difference


def _noise_seed(pair_id: str, initial_state_index: int, replan_index: int, inference_seed: int) -> int:
    payload = f"{pair_id}|{initial_state_index}|{replan_index}|{inference_seed}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _policy_noise(
    pair_id: str,
    initial_state_index: int,
    replan_index: int,
    inference_seed: int,
    action_horizon: int,
    action_dim: int,
) -> np.ndarray:
    rng = np.random.default_rng(_noise_seed(pair_id, initial_state_index, replan_index, inference_seed))
    return rng.standard_normal((action_horizon, action_dim), dtype=np.float32)


def _policy_observation(
    obs: dict,
    prompt: str,
    resize_size: int,
    noise: np.ndarray,
    physical_prompt: PhysicalPrompt | None = None,
    action_order: str = "aligned",
) -> tuple[dict, np.ndarray]:
    image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_image = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    image = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, resize_size, resize_size))
    wrist_image = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_image, resize_size, resize_size))
    element = {
        "observation/image": image,
        "observation/wrist_image": wrist_image,
        "observation/state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        ),
        "prompt": prompt,
        POLICY_NOISE_KEY: noise,
    }
    if physical_prompt is not None:
        prompt_actions = physical_prompt.actions
        if action_order == "reversed":
            prompt_actions = prompt_actions[::-1].copy()
        elif action_order != "aligned":
            raise ValueError(f"Unknown physical-prompt action order: {action_order}")
        element.update(
            {
                "physical_prompt/images": physical_prompt.images,
                "physical_prompt/actions": prompt_actions,
                "physical_prompt/mask": physical_prompt.mask,
            }
        )
        if physical_prompt.post_images is not None:
            element["physical_prompt/post_images"] = physical_prompt.post_images
    return element, image


def _load_physical_prompts(
    dataset_root: str,
    tasks: set[str],
    *,
    num_frames: int,
    seed: int,
    action_horizon: int,
    include_effects: bool = False,
) -> dict[str, PhysicalPrompt]:
    """Load one deterministic other-episode demonstration for each task."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

    from openpi.training.data_loader import DirectParquetLeRobotDataset
    from openpi.training.data_loader import PhysicalPromptDataset

    metadata = LeRobotDatasetMetadata("physical-intelligence/libero", root=dataset_root)
    task_indices = {task: task_index for task_index, task in metadata.tasks.items()}
    missing_tasks = sorted(tasks - set(task_indices))
    if missing_tasks:
        raise ValueError(f"Tasks are absent from the physical-prompt dataset: {missing_tasks}")

    base_dataset = DirectParquetLeRobotDataset(metadata, dataset_root, action_horizon=action_horizon)
    prompt_dataset = PhysicalPromptDataset(
        base_dataset,
        num_frames=num_frames,
        seed=seed,
        include_effects=include_effects,
    )
    prompt_by_task: dict[str, PhysicalPrompt] = {}
    for task in sorted(tasks):
        query_episode = next(
            episode_index for episode_index, episode in metadata.episodes.items() if episode["tasks"] == [task]
        )
        query_index = int(base_dataset.episode_data_index["from"][query_episode])
        item = prompt_dataset[query_index]
        prompt_images = np.asarray(item["physical_prompt_images"])
        prompt_images = np.transpose(prompt_images, (0, 2, 3, 1))
        prompt_images = np.rint(prompt_images * 255).clip(0, 255).astype(np.uint8)
        prompt_post_images = item.get("physical_prompt_post_images")
        if prompt_post_images is not None:
            prompt_post_images = np.asarray(prompt_post_images)
            prompt_post_images = np.transpose(prompt_post_images, (0, 2, 3, 1))
            prompt_post_images = np.rint(prompt_post_images * 255).clip(0, 255).astype(np.uint8)
        prompt_by_task[task] = PhysicalPrompt(
            task=task,
            demo_episode_index=int(item["physical_prompt_episode_index"]),
            images=prompt_images,
            actions=np.asarray(item["physical_prompt_actions"], dtype=np.float32),
            mask=np.asarray(item["physical_prompt_mask"], dtype=bool),
            post_images=prompt_post_images,
        )
    return prompt_by_task


def _prompt_conditions(
    args,
    task_a,
    task_b,
    physical_prompts: dict[str, PhysicalPrompt] | None,
) -> tuple[PromptCondition, ...]:
    if physical_prompts is None:
        return (
            PromptCondition("language_a", task_a.language, "a"),
            PromptCondition("language_b", task_b.language, "b"),
            PromptCondition("neutral_no_information", args.neutral_prompt, None),
        )

    prompt_a = physical_prompts[task_a.language]
    prompt_b = physical_prompts[task_b.language]
    return (
        PromptCondition("physical_a_aligned", args.neutral_prompt, "a", prompt_a),
        PromptCondition("physical_b_aligned", args.neutral_prompt, "b", prompt_b),
        PromptCondition("physical_a_reversed_actions", args.neutral_prompt, "a", prompt_a, "reversed"),
        PromptCondition("physical_b_reversed_actions", args.neutral_prompt, "b", prompt_b, "reversed"),
        PromptCondition("physical_no_prompt", args.neutral_prompt, None),
    )


def validate_pairs(args, task_suites: dict, task_lookups: dict, pair_specs: tuple[PairSpec, ...]) -> None:
    output_path = pathlib.Path(args.output_jsonl) if args.output_jsonl else None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    for spec in pair_specs:
        task_suite = task_suites[spec.suite]
        task_by_name = task_lookups[spec.suite]
        task_id_a, task_a = task_by_name[spec.task_a]
        task_id_b, task_b = task_by_name[spec.task_b]
        bddl_a = _task_bddl_path(task_a).read_text()
        bddl_b = _task_bddl_path(task_b).read_text()
        physics_a = _remove_bddl_sections(bddl_a, ("language", "obj_of_interest", "goal"))
        physics_b = _remove_bddl_sections(bddl_b, ("language", "obj_of_interest", "goal"))
        physics_match = physics_a == physics_b

        initial_states = task_suite.get_task_init_states(task_id_a)
        env_a = _make_env(task_a, args.seed)
        env_b = _make_env(task_b, args.seed)
        try:
            for initial_state_index in args.initial_state_indices:
                obs_a, obs_b, obs_differences, state_difference = _reset_pair(
                    env_a, env_b, initial_states[initial_state_index], args.num_steps_wait
                )
                stabilized_differences = _observation_differences(obs_a, obs_b)
                stabilized_state_difference = _max_state_difference(env_a, env_b)
                carrier_observation = _copy_observation(obs_a)
                repeated_obs_a, _, _, repeated_state_difference = _reset_pair(
                    env_a, env_b, initial_states[initial_state_index], args.num_steps_wait
                )
                carrier_repeat_differences = _observation_differences(carrier_observation, repeated_obs_a)
                row = {
                    "mode": "validate",
                    "suite": spec.suite,
                    "pair_id": spec.pair_id,
                    "task_id_a": task_id_a,
                    "task_id_b": task_id_b,
                    "task_a": task_a.language,
                    "task_b": task_b.language,
                    "initial_state_source_task_id": task_id_a,
                    "initial_state_index": initial_state_index,
                    "physics_bddl_match": physics_match,
                    "physics_bddl_sha256": hashlib.sha256(physics_a.encode()).hexdigest(),
                    "initial_state_max_abs_difference": state_difference,
                    "stabilized_state_max_abs_difference": stabilized_state_difference,
                    "carrier_repeat_state_max_abs_difference": repeated_state_difference,
                    "initial_observation_max_abs_differences": obs_differences,
                    "stabilized_observation_max_abs_differences": stabilized_differences,
                    "carrier_repeat_observation_max_abs_differences": carrier_repeat_differences,
                    "scorer_render_match": bool(
                        obs_differences["agentview_image"] == 0.0
                        and obs_differences["robot0_eye_in_hand_image"] == 0.0
                        and stabilized_differences["agentview_image"] == 0.0
                        and stabilized_differences["robot0_eye_in_hand_image"] == 0.0
                    ),
                    "agentview_sha256": _sha256(carrier_observation["agentview_image"]),
                    "wrist_sha256": _sha256(carrier_observation["robot0_eye_in_hand_image"]),
                    "initial_goal_a": bool(env_a.check_success()),
                    "initial_goal_b": bool(env_b.check_success()),
                }
                row["valid"] = bool(
                    physics_match
                    and state_difference <= STATE_ATOL
                    and stabilized_state_difference <= STATE_ATOL
                    and repeated_state_difference <= STATE_ATOL
                    and all(value == 0.0 for value in carrier_repeat_differences.values())
                    and not row["initial_goal_a"]
                    and not row["initial_goal_b"]
                )
                print(json.dumps(row, sort_keys=True))
                if output_path:
                    with output_path.open("a") as output_file:
                        output_file.write(json.dumps(row, sort_keys=True) + "\n")
        finally:
            env_a.close()
            env_b.close()


def evaluate_pairs(args, task_suites: dict, task_lookups: dict, pair_specs: tuple[PairSpec, ...]) -> None:
    if not args.output_jsonl:
        raise ValueError("--output-jsonl is required in eval mode")
    output_path = pathlib.Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    physical_prompts = None
    if args.physical_prompt_dataset_root:
        physical_prompts = _load_physical_prompts(
            args.physical_prompt_dataset_root,
            {
                task.language
                for spec in pair_specs
                for _, task in (
                    task_lookups[spec.suite][spec.task_a],
                    task_lookups[spec.suite][spec.task_b],
                )
            },
            num_frames=args.physical_prompt_frames,
            seed=args.physical_prompt_seed,
            action_horizon=args.action_horizon,
            include_effects=args.physical_prompt_effects,
        )

    for spec in pair_specs:
        task_suite = task_suites[spec.suite]
        task_by_name = task_lookups[spec.suite]
        task_id_a, task_a = task_by_name[spec.task_a]
        task_id_b, task_b = task_by_name[spec.task_b]
        initial_states = task_suite.get_task_init_states(task_id_a)
        env_a = _make_env(task_a, args.seed)
        env_b = _make_env(task_b, args.seed)
        prompt_conditions = _prompt_conditions(args, task_a, task_b, physical_prompts)
        try:
            for initial_state_index in args.initial_state_indices:
                for condition in prompt_conditions:
                    obs_a, _, initial_obs_differences, initial_state_difference = _reset_pair(
                        env_a, env_b, initial_states[initial_state_index], args.num_steps_wait
                    )
                    action_plan = collections.deque()
                    replay_images: list[np.ndarray] = []
                    replan_index = 0
                    first_goal = None
                    episode_error = None
                    steps_executed = 0
                    max_dynamics_difference = _max_state_difference(env_a, env_b)
                    episode_start = time.monotonic()

                    for _ in range(args.max_steps):
                        try:
                            if not action_plan:
                                noise = _policy_noise(
                                    spec.pair_id,
                                    initial_state_index,
                                    replan_index,
                                    args.inference_seed,
                                    args.action_horizon,
                                    args.action_dim,
                                )
                                element, image = _policy_observation(
                                    obs_a,
                                    condition.language,
                                    args.resize_size,
                                    noise,
                                    condition.physical_prompt,
                                    condition.action_order,
                                )
                                replay_images.append(image)
                                action_chunk = client.infer(element)["actions"]
                                if len(action_chunk) < args.replan_steps:
                                    raise RuntimeError(
                                        f"Policy returned {len(action_chunk)} actions; {args.replan_steps} required"
                                    )
                                action_plan.extend(action_chunk[: args.replan_steps])
                                replan_index += 1

                            action = np.asarray(action_plan.popleft()).tolist()
                            obs_a, _, done_a, _ = env_a.step(action)
                            _, _, done_b, _ = env_b.step(action)
                            steps_executed += 1
                            max_dynamics_difference = max(max_dynamics_difference, _max_state_difference(env_a, env_b))
                            if done_a or done_b:
                                first_goal = "both" if done_a and done_b else ("a" if done_a else "b")
                                break
                        except Exception as error:  # Keep an exact per-episode failure record.
                            episode_error = repr(error)
                            break

                    success = condition.expected_goal is not None and first_goal == condition.expected_goal
                    video_path = None
                    if args.save_video and replay_images:
                        import imageio

                        suffix = first_goal or "none"
                        video_path = pathlib.Path(args.video_out_path) / (
                            f"{spec.pair_id}_init{initial_state_index}_{condition.name}_{suffix}.mp4"
                        )
                        video_path.parent.mkdir(parents=True, exist_ok=True)
                        imageio.mimwrite(video_path, replay_images, fps=10)

                    row = {
                        "mode": "eval",
                        "suite": spec.suite,
                        "pair_id": spec.pair_id,
                        "task_id_a": task_id_a,
                        "task_id_b": task_id_b,
                        "task_a": task_a.language,
                        "task_b": task_b.language,
                        "initial_state_source_task_id": task_id_a,
                        "initial_state_index": initial_state_index,
                        "environment_seed": args.seed,
                        "inference_seed": args.inference_seed,
                        "condition_name": condition.name,
                        "policy_prompt": condition.language,
                        "physical_prompt": (
                            condition.physical_prompt.metadata(action_order=condition.action_order)
                            if condition.physical_prompt is not None
                            else None
                        ),
                        "expected_goal": condition.expected_goal,
                        "first_goal": first_goal,
                        "success": bool(success),
                        "steps": steps_executed,
                        "replans": replan_index,
                        "wall_time_seconds": time.monotonic() - episode_start,
                        "initial_state_max_abs_difference": initial_state_difference,
                        "initial_observation_max_abs_differences": initial_obs_differences,
                        "max_dynamics_state_difference": max_dynamics_difference,
                        "error": episode_error,
                        "video": str(video_path) if video_path else None,
                    }
                    with output_path.open("a") as output_file:
                        output_file.write(json.dumps(row, sort_keys=True) + "\n")
                    print(json.dumps(row, sort_keys=True))
        finally:
            env_a.close()
            env_b.close()


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.array(quat, copy=True)
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(denominator, 0.0):
        return np.zeros(3)
    return quat[:3] * 2.0 * math.acos(quat[3]) / denominator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("validate", "eval"), default="validate")
    parser.add_argument("--pairs", type=_parse_csv, default=tuple(PAIR_SPECS))
    parser.add_argument("--initial-state-indices", type=_parse_int_csv, default=(0, 1, 2, 3, 4))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--inference-seed", type=int, default=0)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--action-dim", type=int, default=32)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--neutral-prompt", default=NEUTRAL_PROMPT)
    parser.add_argument(
        "--physical-prompt-dataset-root",
        help="Enable sensorimotor-prompt conditions using this local LeRobot dataset root.",
    )
    parser.add_argument("--physical-prompt-frames", type=int, default=8)
    parser.add_argument("--physical-prompt-seed", type=int, default=42)
    parser.add_argument(
        "--physical-prompt-effects",
        action="store_true",
        help="Send post-action images for an effect-binding physical-prompt model.",
    )
    parser.add_argument("--output-jsonl")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-out-path", default="/tmp/libero_counterfactual_videos")
    args = parser.parse_args()

    unknown_pairs = sorted(set(args.pairs) - set(PAIR_SPECS))
    if unknown_pairs:
        raise ValueError(f"Unknown pair IDs: {unknown_pairs}")
    pair_specs = tuple(PAIR_SPECS[pair_id] for pair_id in args.pairs)
    benchmark_dict = benchmark.get_benchmark_dict()
    suites = sorted({spec.suite for spec in pair_specs})
    task_suites = {suite: benchmark_dict[suite]() for suite in suites}
    task_lookups = {suite: _task_lookup(task_suite) for suite, task_suite in task_suites.items()}
    if args.mode == "validate":
        validate_pairs(args, task_suites, task_lookups, pair_specs)
    else:
        evaluate_pairs(args, task_suites, task_lookups, pair_specs)


if __name__ == "__main__":
    main()
