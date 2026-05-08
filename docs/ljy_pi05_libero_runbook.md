# Ljy pi05 LIBERO 新手运行手册

本文档面向第一次在服务器上跑 `pi05_libero` 的新手。目标顺序是：

1. 先确认本地和服务器都准备好；
2. 再安装环境；
3. 再找一张空闲 GPU；
4. 先跑最小 inference；
5. 最后再跑 LIBERO 仿真。

除非特别说明，下面所有服务器命令都默认在这个目录运行：

```bash
cd /home/csuvla/ljy/projects/openpi-ljy
```

凡是会用 GPU 的命令，都必须手动指定空闲卡号：

```bash
CUDA_VISIBLE_DEVICES=空闲卡号 ...
```

例如你检查后发现 2 号卡空闲，就把命令里的 `空闲卡号` 替换成 `2`。

## 1. 本地准备

本地电脑主要负责连服务器、传文件、看日志，不负责跑模型。

### 1.1 准备 SSH

在本地终端执行：

```bash
ssh csuvla@<服务器IP或域名>
```

如果能登录，说明 SSH 没问题。登录后先进入项目目录：

```bash
cd /home/csuvla/ljy/projects/openpi-ljy
pwd
```

期望输出是：

```text
/home/csuvla/ljy/projects/openpi-ljy
```

### 1.2 建议使用 tmux

模型服务和 LIBERO 仿真可能跑很久。建议在服务器上开 `tmux`，这样断开 SSH 后任务不会立刻退出。

```bash
cd /home/csuvla/ljy/projects/openpi-ljy
tmux new -s pi05-libero
```

常用快捷键：

- 退出但不停止任务：按 `Ctrl-b`，再按 `d`
- 重新进入：`tmux attach -t pi05-libero`
- 查看已有会话：`tmux ls`

## 2. 服务器准备

### 2.1 确认项目目录

```bash
cd /home/csuvla/ljy/projects/openpi-ljy
git status --short
```

如果只是想跑 inference，不需要改源码，也不需要提交代码。

### 2.2 初始化子模块

LIBERO 示例依赖 git submodule。

**会下载大文件/大量文件：** 下面命令可能会从网络拉取第三方仓库。

```bash
cd /home/csuvla/ljy/projects/openpi-ljy
git submodule update --init --recursive
```

### 2.3 检查基础工具

```bash
cd /home/csuvla/ljy/projects/openpi-ljy
python3 --version
uv --version
git --version
nvidia-smi
```

这个项目要求 Python `>=3.11`。如果 `uv` 不存在，需要先安装 `uv`，或让管理员确认服务器上已有可用的 `uv`。

## 3. 环境安装

### 3.1 安装 openpi Python 环境

**会下载大文件/大量文件：** 下面命令会下载 Python 依赖、JAX CUDA 包、PyTorch、LeRobot 等依赖，首次执行会比较慢。

```bash
cd /home/csuvla/ljy/projects/openpi-ljy
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

说明：

- `GIT_LFS_SKIP_SMUDGE=1` 是项目 README 要求的设置，用来避免依赖安装时拉取不必要的 LFS 大文件。
- 如果网络中断，通常可以重新执行同一条命令。

### 3.2 安装 LIBERO 仿真依赖

官方推荐用 Docker 跑 LIBERO，因为 Mujoco、OpenGL、LIBERO 依赖比较容易冲突。

如果走 Docker 路线，先确认 Docker 和 NVIDIA container toolkit 可用：

```bash
cd /home/csuvla/ljy/projects/openpi-ljy
docker --version
docker compose version
CUDA_VISIBLE_DEVICES=空闲卡号 docker run --rm --gpus '"device=空闲卡号"' nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

**会下载大文件：** 如果本机没有 `nvidia/cuda:12.4.1-base-ubuntu22.04` 镜像，上面的 `docker run` 会拉取 CUDA Docker 镜像。

如果 Docker 不可用，再考虑非 Docker 路线。非 Docker 路线需要 Python 3.8 的 LIBERO 环境，容易和 openpi 主环境冲突，新手不建议优先走这条。

## 4. GPU 检查

### 4.1 找空闲 GPU

```bash
cd /home/csuvla/ljy/projects/openpi-ljy
nvidia-smi
```

看两件事：

- `Memory-Usage` 越低越可能空闲；
- `Processes` 里没有别人正在用的卡更适合。

假设你看到 2 号卡空闲，后面所有 GPU 命令都写成：

```bash
CUDA_VISIBLE_DEVICES=空闲卡号 ...
```

不要直接写：

```bash
uv run ...
```

因为这样可能占用默认 GPU，影响别人。

### 4.2 用指定 GPU 检查 JAX

```bash
cd /home/csuvla/ljy/projects/openpi-ljy
CUDA_VISIBLE_DEVICES=空闲卡号 uv run python -c "import jax; print(jax.devices())"
```

如果输出里能看到 `cuda` 或 `gpu`，说明 JAX 能看到这张卡。

## 5. 最小 inference

这一节先不跑 LIBERO 仿真，只验证 `pi05_libero` policy server 能启动，并且 simple client 能请求一次模型。

需要两个终端，建议两个 tmux pane 或两个 SSH 窗口。

### 5.1 终端 1：启动 policy server

**会下载大文件：** 第一次启动会自动下载 `gs://openpi-assets/checkpoints/pi05_libero` 到缓存目录，模型 checkpoint 很大。

```bash
cd /home/csuvla/ljy/projects/openpi-ljy
CUDA_VISIBLE_DEVICES=空闲卡号 uv run scripts/serve_policy.py --env LIBERO
```

看到类似 server listening、port `8000`、checkpoint restore 的日志后，不要关闭这个终端。

如果想明确指定 checkpoint，也可以用：

```bash
cd /home/csuvla/ljy/projects/openpi-ljy
CUDA_VISIBLE_DEVICES=空闲卡号 uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=gs://openpi-assets/checkpoints/pi05_libero
```

### 5.2 终端 2：运行 simple client

simple client 会构造随机观测并请求 server，不需要真实机器人。

```bash
cd /home/csuvla/ljy/projects/openpi-ljy
uv run examples/simple_client/main.py --env LIBERO
```

如果能看到 inference rate 或 action 返回相关日志，说明最小 inference 已跑通。

### 5.3 停止 server

回到终端 1，按：

```text
Ctrl-c
```

## 6. LIBERO 仿真

LIBERO 仿真建议使用 Docker。它会同时跑 policy server 和 LIBERO evaluation client。

### 6.1 允许 Docker 使用显示服务

如果服务器有图形显示/X11，先执行：

```bash
cd /home/csuvla/ljy/projects/openpi-ljy
sudo xhost +local:docker
```

如果你没有 sudo 权限，或服务器是纯命令行环境，先跳过这条，遇到显示/OpenGL 报错再看“常见报错排查”。

### 6.2 运行默认 LIBERO 评测

**会下载大文件：** 首次运行会构建 Docker 镜像、下载 apt/pip 依赖，并可能下载 `pi05_libero` checkpoint。

```bash
cd /home/csuvla/ljy/projects/openpi-ljy
CUDA_VISIBLE_DEVICES=空闲卡号 SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

如果遇到 EGL/Mujoco 显示问题，改用 GLX：

```bash
cd /home/csuvla/ljy/projects/openpi-ljy
CUDA_VISIBLE_DEVICES=空闲卡号 MUJOCO_GL=glx SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

### 6.3 指定 LIBERO task suite

例如跑 `libero_10`：

**会下载大文件：** 如果镜像或 checkpoint 尚未缓存，仍会下载。

```bash
cd /home/csuvla/ljy/projects/openpi-ljy
CUDA_VISIBLE_DEVICES=空闲卡号 SERVER_ARGS="--env LIBERO" CLIENT_ARGS="--args.task-suite-name libero_10" docker compose -f examples/libero/compose.yml up --build
```

## 7. 日志保存

建议每次实验都把日志保存到 `logs/`，方便之后排查。

### 7.1 创建日志目录

```bash
cd /home/csuvla/ljy/projects/openpi-ljy
mkdir -p logs
```

### 7.2 保存最小 inference 的 server 日志

**会下载大文件：** 如果 checkpoint 还没缓存，这条命令会下载 `pi05_libero` checkpoint。

```bash
cd /home/csuvla/ljy/projects/openpi-ljy
CUDA_VISIBLE_DEVICES=空闲卡号 uv run scripts/serve_policy.py --env LIBERO 2>&1 | tee logs/pi05_libero_server.log
```

另一个终端运行 client：

```bash
cd /home/csuvla/ljy/projects/openpi-ljy
uv run examples/simple_client/main.py --env LIBERO 2>&1 | tee logs/pi05_libero_simple_client.log
```

### 7.3 保存 LIBERO Docker 日志

**会下载大文件：** 如果镜像或 checkpoint 尚未缓存，仍会下载。

```bash
cd /home/csuvla/ljy/projects/openpi-ljy
CUDA_VISIBLE_DEVICES=空闲卡号 SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build 2>&1 | tee logs/pi05_libero_docker_eval.log
```

后台运行时可以用：

```bash
cd /home/csuvla/ljy/projects/openpi-ljy
CUDA_VISIBLE_DEVICES=空闲卡号 SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build -d
docker compose -f examples/libero/compose.yml logs -f | tee logs/pi05_libero_docker_eval.log
```

停止后台服务：

```bash
cd /home/csuvla/ljy/projects/openpi-ljy
docker compose -f examples/libero/compose.yml down
```

## 8. 常见报错排查

### 8.1 `uv: command not found`

说明服务器没有安装 `uv`，或者 `uv` 不在 `PATH` 里。

先检查：

```bash
cd /home/csuvla/ljy/projects/openpi-ljy
which uv
```

如果没有输出，需要安装 `uv` 或让管理员配置。

### 8.2 Python 版本不对

项目主环境要求 Python `>=3.11`。

检查：

```bash
cd /home/csuvla/ljy/projects/openpi-ljy
uv run python --version
```

如果低于 3.11，重新创建/同步 uv 环境：

**会下载大文件/大量文件：** 会重新解析和安装依赖。

```bash
cd /home/csuvla/ljy/projects/openpi-ljy
uv venv --python 3.11
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### 8.3 `jax` 看不到 GPU

先确认你没有忘记指定 GPU：

```bash
cd /home/csuvla/ljy/projects/openpi-ljy
CUDA_VISIBLE_DEVICES=空闲卡号 uv run python -c "import jax; print(jax.devices())"
```

如果只看到 CPU：

- 确认 `nvidia-smi` 正常；
- 确认 NVIDIA driver 支持当前 CUDA/JAX；
- 重新执行环境安装；
- 不要在没有 GPU 的登录节点上跑模型。

### 8.4 GPU 显存不够 / OOM

现象通常是 `CUDA out of memory`、`RESOURCE_EXHAUSTED`、`failed to allocate`。

处理：

```bash
cd /home/csuvla/ljy/projects/openpi-ljy
nvidia-smi
```

换一张空闲卡，例如把 `空闲卡号` 替换成另一张空闲卡：

```bash
cd /home/csuvla/ljy/projects/openpi-ljy
CUDA_VISIBLE_DEVICES=空闲卡号 uv run scripts/serve_policy.py --env LIBERO
```

不要和别人共用一张已经占满的卡。

### 8.5 checkpoint 下载失败

`pi05_libero` 默认从：

```text
gs://openpi-assets/checkpoints/pi05_libero
```

下载并缓存到：

```text
~/.cache/openpi
```

如果下载失败：

- 检查服务器网络；
- 重新执行同一条启动命令；
- 检查磁盘空间：`df -h`；
- 如果默认 home 空间不够，可以设置 `OPENPI_DATA_HOME` 到大容量磁盘。

示例：

**会下载大文件：** 会把 checkpoint 缓存到指定目录。

```bash
cd /home/csuvla/ljy/projects/openpi-ljy
mkdir -p /home/csuvla/ljy/openpi_cache
OPENPI_DATA_HOME=/home/csuvla/ljy/openpi_cache CUDA_VISIBLE_DEVICES=空闲卡号 uv run scripts/serve_policy.py --env LIBERO
```

### 8.6 Docker 不能用 GPU

检查：

```bash
cd /home/csuvla/ljy/projects/openpi-ljy
CUDA_VISIBLE_DEVICES=空闲卡号 docker run --rm --gpus '"device=空闲卡号"' nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

**会下载大文件：** 如果 CUDA 镜像不存在，会先下载镜像。

如果失败，通常是 NVIDIA container toolkit 没装好，或 Docker 不是 rootless/配置不兼容。需要管理员处理 Docker GPU 配置。

### 8.7 Mujoco / EGL / OpenGL 报错

先尝试 GLX：

**会下载大文件：** 如果镜像或 checkpoint 尚未缓存，仍会下载。

```bash
cd /home/csuvla/ljy/projects/openpi-ljy
CUDA_VISIBLE_DEVICES=空闲卡号 MUJOCO_GL=glx SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

如果仍失败：

- 确认是否有可用显示环境；
- 确认是否执行过 `sudo xhost +local:docker`；
- 保存完整日志到 `logs/pi05_libero_docker_eval.log` 后再排查。

### 8.8 端口 8000 被占用

policy server 默认使用 8000 端口。

检查：

```bash
cd /home/csuvla/ljy/projects/openpi-ljy
lsof -i :8000
```

如果是你自己之前启动的进程，可以停止它。也可以换端口启动 server：

**会下载大文件：** 如果 checkpoint 还没缓存，这条命令会下载 checkpoint。

```bash
cd /home/csuvla/ljy/projects/openpi-ljy
CUDA_VISIBLE_DEVICES=空闲卡号 uv run scripts/serve_policy.py --env LIBERO --port 8001
```

注意 client 也要连接同一个端口；simple client 默认通常连 8000，改端口前先看：

```bash
cd /home/csuvla/ljy/projects/openpi-ljy
uv run examples/simple_client/main.py --help
```

## 9. 可选阶段：fine-tune

新手不要把 full fine-tune 当第一步。建议顺序是：

1. 先跑通最小 inference；
2. 再跑通 LIBERO Docker 仿真；
3. 再考虑 LoRA 或小规模 fine-tune；
4. 最后才考虑 full fine-tune。

### 9.1 计算 LIBERO norm stats

**会下载大文件：** 这个命令可能会下载/读取 LIBERO 训练数据和相关 assets。

```bash
cd /home/csuvla/ljy/projects/openpi-ljy
CUDA_VISIBLE_DEVICES=空闲卡号 uv run scripts/compute_norm_stats.py --config-name pi05_libero
```

### 9.2 启动 full fine-tune

**会下载大文件，并且会长时间占用 GPU：** 会加载 base checkpoint、读取数据、写 checkpoint。full fine-tune 对显存要求很高，项目 README 估计需要 70GB 以上显存，适合 A100 80GB / H100 这类卡。

```bash
cd /home/csuvla/ljy/projects/openpi-ljy
CUDA_VISIBLE_DEVICES=空闲卡号 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=ljy_pi05_libero_full --overwrite
```

训练 checkpoint 默认保存在：

```text
checkpoints/pi05_libero/ljy_pi05_libero_full/
```

训练完成后，用某个 step 的 checkpoint 启动 inference，例如：

```bash
cd /home/csuvla/ljy/projects/openpi-ljy
CUDA_VISIBLE_DEVICES=空闲卡号 uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=checkpoints/pi05_libero/ljy_pi05_libero_full/20000
```
