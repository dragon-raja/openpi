# π0.5 LIBERO Baseline Log

## Environment

- Lab: HIT Shenzhen
- User: longjunyu
- Repo: openpi-ljy
- Branch: hitshenzhen-pi05-libero
- Policy: pi05_libero
- Checkpoint: `gs://openpi-assets/checkpoints/pi05_libero`
- Server: `scripts/serve_policy.py --env LIBERO`
- Eval: `examples/libero/main.py`
- Server env: `.venv`
- Eval env: `examples/libero/.venv`
- Code root: `/data/longjunyu/projects/openpi-ljy` or `/home/longjunyu/projects/openpi-ljy`
- Result root: `/nas/longjunyu/results/pi05_libero`
- Cache root: `/nas/longjunyu/cache`

## Notes

- Code and virtual environments should stay on server local disk, not NAS.
- NAS is used for checkpoint cache, logs, videos, datasets, and baseline results.
- Use GitHub to sync code between Windows workstation and servers.
- Use VS Code Remote-SSH to operate servers.
- Use Codex App mainly on the Windows local repo for code reading, documentation, and small patches.

## Results

| Date | Host | GPU | Suite | Trials/task | Tasks | Episodes | Successes | Success rate | Log path | Notes |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| 2026-xx-xx | expm10 | 4090-0 | libero_spatial | 1 | 10 | ? | ? | ? | `/nas/longjunyu/results/pi05_libero/logs/...log` | smoke eval |

## Command Templates

### Server

```bash
cd $PI05_REPO
source /home/longjunyu/scripts/env_pi05.sh
CUDA_VISIBLE_DEVICES=GPU_ID uv run python scripts/serve_policy.py --env LIBERO
```

### Simple Client

```bash
cd $PI05_REPO
source /home/longjunyu/scripts/env_pi05.sh
uv run python examples/simple_client/main.py --env LIBERO --num-steps 3
```

### LIBERO Eval

```bash
cd $PI05_REPO
source /home/longjunyu/scripts/env_pi05.sh
source examples/libero/.venv/bin/activate

CUDA_VISIBLE_DEVICES=GPU_ID MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python examples/libero/main.py \
  --args.task-suite-name SUITE \
  --args.num-trials-per-task N
```

## Baseline Plan

### Round 1: smoke baseline

- `libero_spatial`, trials/task = 1
- `libero_object`, trials/task = 1
- `libero_goal`, trials/task = 1
- `libero_10`, trials/task = 1

### Round 2: more stable baseline

- `libero_spatial`, trials/task = 3
- `libero_object`, trials/task = 3
- `libero_goal`, trials/task = 3
- `libero_10`, trials/task = 3

### Round 3: optional formal baseline

- trials/task = 10

## Concepts

- benchmark = LIBERO
- suite = `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`
- task = one instruction/environment
- trial / episode / rollout = one full attempt
- step = one env action loop
- success rate = successful episodes / total episodes
- action_chunk = model predicts multiple future actions
- replan_steps = execute first N actions before querying server again

## Common Issues

### Submodule

Check:

```bash
git submodule status
```

Healthy state: `third_party/aloha` and `third_party/libero` should not start with `-` or `+`.

### LIBERO import failure

Fix by writing a `.pth` file into `examples/libero/.venv`.

### Websocket proxy issue

Local client/eval connects to local server. Do not route local websocket through proxy.

### EGL cleanup warning

If success rate and total episodes have already printed, EGL cleanup warning at program exit can usually be ignored.

### NAS

Do not put `.venv` or the whole code runtime on NAS.
