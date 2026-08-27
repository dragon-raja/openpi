"""Episode-aware evaluation for physical-prompt behavior binding."""

import argparse
import dataclasses
import json
import pathlib

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import train

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader

TASKS = (
    ("heldout", "put both the alphabet soup and the tomato sauce in the basket"),
    ("heldout", "put both the cream cheese box and the butter in the basket"),
    ("trained", "put the bowl on the plate"),
    ("trained", "put the bowl on the stove"),
    ("trained", "put the wine bottle on the rack"),
    ("trained", "put the wine bottle on top of the cabinet"),
)
PHASES = (0.2, 0.4, 0.6, 0.8)
METRICS = ("three_way_correct", "wrong_pair_correct", "order_correct", "wrong_gap", "reversed_gap")


def _build_queries(config: _config.TrainConfig, episodes_per_task: int):
    data_factory = dataclasses.replace(config.data, physical_prompt_language_anchor_fraction=0.0)
    data_config = data_factory.create(config.assets_dirs, config.model)
    raw_dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    metadata = _data_loader._dataset_metadata(raw_dataset)  # noqa: SLF001
    episode_index = _data_loader._episode_data_index(raw_dataset)  # noqa: SLF001
    if metadata is None or episode_index is None:
        raise RuntimeError("LIBERO metadata and episode boundaries are required")
    starts = np.asarray(episode_index["from"], dtype=np.int64)
    ends = np.asarray(episode_index["to"], dtype=np.int64)

    indices = []
    query_metadata = []
    for split, task in TASKS:
        episodes = [episode for episode in sorted(metadata.episodes) if metadata.episodes[episode]["tasks"] == [task]][
            :episodes_per_task
        ]
        if len(episodes) != episodes_per_task:
            raise ValueError(f"Task {task!r} has only {len(episodes)} episodes")
        for episode in episodes:
            usable = int(ends[episode] - starts[episode]) - config.model.action_horizon
            for phase_index, phase in enumerate(PHASES):
                local_index = int(round((usable - 1) * phase))
                indices.append(int(starts[episode] + local_index))
                query_metadata.append(
                    {
                        "split": split,
                        "task": task,
                        "episode": int(episode),
                        "phase_index": phase_index,
                        "phase_fraction": phase,
                        "global_index": indices[-1],
                    }
                )
    dataset = _data_loader.transform_dataset(raw_dataset, data_config)
    return dataset, indices, query_metadata


def _load_model(config: _config.TrainConfig, checkpoint: pathlib.Path):
    params = _model.restore_params(checkpoint / "params")
    model = config.model.load(params)
    model.eval()
    return model


def _score_batch(model, observation, actions, stage_temperature):
    query = model.encode_query_action_behavior(actions, observation)
    candidates = (
        observation,
        train._intervene_physical_prompt(observation, use_counterfactual=True),  # noqa: SLF001
        train._intervene_physical_prompt(observation, reverse_actions=True),  # noqa: SLF001
    )
    if model.physical_prompt_stage_alignment:
        encoded = [model.encode_physical_prompt_behavior_stages(candidate) for candidate in candidates]
        scores = [
            model.score_query_prompt_behavior(query, stages, mask, temperature=stage_temperature)
            for stages, mask in encoded
        ]
        positive_stages, positive_mask = encoded[0]
        stage_similarity = jnp.einsum("bd,bpd->bp", query, positive_stages)
        top_stage = jnp.argmax(jnp.where(positive_mask, stage_similarity, -jnp.inf), axis=-1)
    else:
        encoded = [model.encode_physical_prompt_behavior(candidate) for candidate in candidates]
        scores = [jnp.sum(query * candidate, axis=-1) for candidate in encoded]
        top_stage = jnp.full(query.shape[0], -1, dtype=jnp.int32)
    return jnp.stack(scores, axis=-1), top_stage


def _aggregate(records):
    result = {"queries": len(records), "episodes": len({(r["task"], r["episode"]) for r in records})}
    for metric in METRICS:
        result[metric.replace("_correct", "_accuracy")] = float(np.mean([r[metric] for r in records]))
    top_stages = [r["top_stage"] for r in records if r["top_stage"] >= 0]
    if top_stages:
        counts = np.bincount(top_stages, minlength=8)
        probabilities = counts[counts > 0] / counts.sum()
        result["top_stage_counts"] = counts.tolist()
        result["top_stage_entropy_normalized"] = float(
            -np.sum(probabilities * np.log(probabilities)) / np.log(len(counts))
        )
    return result


def _stratified_episode_bootstrap(records, *, samples: int, seed: int):
    rng = np.random.default_rng(seed)
    task_names = sorted({record["task"] for record in records})
    by_task_episode = {}
    for task in task_names:
        task_records = [record for record in records if record["task"] == task]
        episode_ids = sorted({record["episode"] for record in task_records})
        by_task_episode[task] = {
            episode: [record for record in task_records if record["episode"] == episode] for episode in episode_ids
        }

    distributions = {metric: np.empty(samples, dtype=np.float64) for metric in METRICS}
    for sample in range(samples):
        task_values = {metric: [] for metric in METRICS}
        for task in task_names:
            episode_map = by_task_episode[task]
            episode_ids = np.asarray(sorted(episode_map))
            selected = rng.choice(episode_ids, size=len(episode_ids), replace=True)
            for metric in METRICS:
                task_values[metric].append(
                    np.mean([record[metric] for episode in selected for record in episode_map[int(episode)]])
                )
        for metric in METRICS:
            distributions[metric][sample] = np.mean(task_values[metric])

    intervals = {}
    for metric, distribution in distributions.items():
        intervals[metric.replace("_correct", "_accuracy")] = {
            "ci90": np.quantile(distribution, [0.05, 0.95]).tolist(),
            "ci95": np.quantile(distribution, [0.025, 0.975]).tolist(),
        }
    return intervals


def _paired_bootstrap(records, baseline_payload, *, samples: int, seed: int):
    baseline_by_key = {
        (record["task"], record["episode"], record["phase_index"]): record for record in baseline_payload["records"]
    }
    deltas = []
    for record in records:
        key = (record["task"], record["episode"], record["phase_index"])
        if key not in baseline_by_key:
            raise ValueError(f"Baseline is missing paired query {key}")
        baseline = baseline_by_key[key]
        deltas.append(
            {
                **{name: record[name] for name in ("split", "task", "episode", "phase_index")},
                **{metric: record[metric] - baseline[metric] for metric in METRICS},
                "top_stage": -1,
            }
        )
    return {
        split: {
            "point": _aggregate([record for record in deltas if record["split"] == split]),
            "bootstrap": _stratified_episode_bootstrap(
                [record for record in deltas if record["split"] == split], samples=samples, seed=seed
            ),
        }
        for split in ("trained", "heldout")
    }


def main(args):
    config = _config.get_config(args.config)
    dataset, indices, query_metadata = _build_queries(config, args.episodes_per_task)
    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=args.batch_size,
        sampler=indices,
        num_batches=len(indices) // args.batch_size,
        num_workers=0,
        framework="jax",
    )
    model = _load_model(config, args.checkpoint)
    graphdef, state = nnx.split(model)

    @jax.jit
    def score(state, observation, actions):
        return _score_batch(nnx.merge(graphdef, state), observation, actions, args.stage_temperature)

    records = []
    offset = 0
    for batch in loader:
        observation = _model.Observation.from_dict(batch)
        scores, top_stage = score(state, observation, batch["actions"])
        scores = np.asarray(jax.device_get(scores))
        top_stage = np.asarray(jax.device_get(top_stage))
        for row in range(scores.shape[0]):
            positive, wrong, reversed_score = (float(value) for value in scores[row])
            records.append(
                {
                    **query_metadata[offset + row],
                    "positive_score": positive,
                    "wrong_score": wrong,
                    "reversed_score": reversed_score,
                    "wrong_gap": positive - wrong,
                    "reversed_gap": positive - reversed_score,
                    "three_way_correct": float(positive > max(wrong, reversed_score)),
                    "wrong_pair_correct": float(positive > wrong),
                    "order_correct": float(positive > reversed_score),
                    "top_stage": int(top_stage[row]),
                }
            )
        offset += scores.shape[0]
    if offset != len(query_metadata):
        raise RuntimeError(f"Expected {len(query_metadata)} queries, evaluated {offset}")

    tasks = {task: _aggregate([record for record in records if record["task"] == task]) for _, task in TASKS}
    groups = {}
    for split in ("trained", "heldout"):
        split_records = [record for record in records if record["split"] == split]
        groups[split] = {
            **_aggregate(split_records),
            "bootstrap": _stratified_episode_bootstrap(
                split_records,
                samples=args.bootstrap_samples,
                seed=args.seed,
            ),
        }
    payload = {
        "schema_version": 1,
        "config": args.config,
        "checkpoint": str(args.checkpoint),
        "episodes_per_task": args.episodes_per_task,
        "phases": PHASES,
        "tasks": tasks,
        "groups": groups,
        "records": records,
    }
    if args.baseline_json is not None:
        with args.baseline_json.open() as file:
            baseline_payload = json.load(file)
        payload["paired_vs_baseline"] = _paired_bootstrap(
            records,
            baseline_payload,
            samples=args.bootstrap_samples,
            seed=args.seed,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
    for task, result in tasks.items():
        print("TASK_RESULT", json.dumps({"task": task, **result}, sort_keys=True), flush=True)
    print("FINAL_RESULT", json.dumps({key: value for key, value in payload.items() if key != "records"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--baseline-json", type=pathlib.Path)
    parser.add_argument("--episodes-per-task", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--stage-temperature", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
