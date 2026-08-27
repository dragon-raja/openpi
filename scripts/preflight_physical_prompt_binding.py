"""Check that a physical-prompt binding objective can learn one fixed real batch.

This is an implementation gate, not an evaluation: it loads the released
checkpoint, finds a batch containing strict ranking examples, and repeatedly
optimizes that same batch without writing a checkpoint.
"""

import dataclasses
import functools
import json

import jax
import jax.numpy as jnp
import train

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding


def main(config_name: str, *, steps: int = 12) -> None:
    base_config = _config.get_config(config_name)
    config = dataclasses.replace(
        base_config,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1,
            peak_lr=3e-5,
            decay_steps=steps,
            decay_lr=3e-5,
        ),
        physical_prompt_behavior_bind_weight=1.0,
        physical_prompt_behavior_bind_warmup_steps=0,
        physical_prompt_behavior_bind_ramp_steps=0,
        wandb_enabled=False,
    )
    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    data_loader = _data_loader.create_data_loader(config, sharding=data_sharding, shuffle=True)

    batch = None
    for candidate in data_loader:
        rank_mask = candidate[0].physical_prompt_rank_mask
        if rank_mask is not None and bool(jax.device_get(jnp.any(rank_mask))):
            batch = candidate
            break
    if batch is None:
        raise RuntimeError("No strict physical-prompt ranking batch was found")

    state, state_sharding = train.init_train_state(config, init_rng, mesh, resume=False)
    step_fn = jax.jit(
        functools.partial(train.train_step, config),
        in_shardings=(replicated_sharding, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    records = []
    for _ in range(steps):
        with sharding.set_mesh(mesh):
            state, info = step_fn(train_rng, state, batch)
        record = {
            "step": int(jax.device_get(state.step)) - 1,
            **{
                name: float(jax.device_get(info[name]))
                for name in (
                    "behavior_bind_loss",
                    "behavior_bind_accuracy",
                    "behavior_bind_wrong_gap",
                    "behavior_bind_reversed_gap",
                    "behavior_bind_grad_norm",
                    "physical_prompt_rank_valid_fraction",
                )
            },
        }
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    first = records[0]
    last = records[-1]
    summary = {
        "config": config_name,
        "steps": steps,
        "loss_delta": last["behavior_bind_loss"] - first["behavior_bind_loss"],
        "accuracy_delta": last["behavior_bind_accuracy"] - first["behavior_bind_accuracy"],
        "wrong_gap_delta": last["behavior_bind_wrong_gap"] - first["behavior_bind_wrong_gap"],
        "reversed_gap_delta": last["behavior_bind_reversed_gap"] - first["behavior_bind_reversed_gap"],
        "all_finite": all(all(jnp.isfinite(value) for value in record.values()) for record in records),
    }
    print(json.dumps({"summary": summary}, sort_keys=True), flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pi05_libero_stage_aligned_behavior_binding_lora")
    parser.add_argument("--steps", type=int, default=12)
    args = parser.parse_args()
    main(args.config, steps=args.steps)
