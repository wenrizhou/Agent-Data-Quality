# SGLang 推理服务脚本

本目录负责启动 judge 模型的 SGLang 推理服务，供 `data_quality/judge` 打分流程调用。

## 背景：整体工作流

```
[这个目录]              [judge 目录]
serve_sglang.sh  →  SGLang HTTP 服务  ←  judge/run_judge.py
   启动模型              端口 31877          发送打分请求
```

judge runner 本身**不部署模型**，它只是一个 HTTP 客户端。因此跑打分之前，必须先用本目录的脚本把模型服务跑起来，再在另一个终端执行 `run_judge.py`。

## 快速上手

**场景一：本地/容器，前台运行（推荐调试时用）**

```bash
bash data_quality/scripts/serve/serve_sglang.sh \
  data_quality/scripts/serve/models/qwen3-32b.sh \
  --wait
```

`--wait` 会阻塞直到服务健康检查通过，再返回控制权。

**场景二：本地/容器，后台运行**

```bash
bash data_quality/scripts/serve/serve_sglang.sh \
  data_quality/scripts/serve/models/qwen3-32b.sh \
  --background --wait
```

服务在后台运行，`--wait` 确认启动成功后脚本退出，可直接开始提交打分任务。

## 本地 / 容器用法

### 启动服务

```bash
bash data_quality/scripts/serve/serve_sglang.sh \
  data_quality/scripts/serve/models/<preset>.sh \
  [选项] [-- <SGLang 额外参数>]
```

**常用选项：**

| 选项 | 说明 |
|------|------|
| `--wait` | 等待服务就绪再返回 |
| `--background` | 后台运行（配合 `--wait` 使用） |
| `--model /path/to/model` | 覆盖 preset 中的模型路径 |
| `--model-name <name>` | 覆盖模型服务名 |
| `--port 31877` | 覆盖端口 |
| `--tp 2` | 覆盖 tensor parallel 大小 |
| `--dp 1` | 覆盖 data parallel 大小 |

传递额外 SGLang 参数（放在 `--` 之后）：

```bash
bash data_quality/scripts/serve/serve_sglang.sh \
  data_quality/scripts/serve/models/qwen3-32b.sh \
  --wait -- --max-total-tokens 65536
```

### 停止服务

```bash
bash data_quality/scripts/serve/stop_sglang.sh
```

## SLURM 用法

```bash
mkdir -p data_quality/logs/serve

sbatch data_quality/scripts/serve/submit_sglang.sh \
  data_quality/scripts/serve/models/qwen3-32b.sh \
  --wait
```

服务跑在 SLURM 节点上时，需要把节点 IP 和端口写入 judge 配置文件：

```yaml
# judge/configs/judge_all_metrics.yaml
client:
  host: <slurm_node_ip>
  port: 31877
  endpoint: /generate
```

节点 IP 可以从 SLURM job 日志或 `squeue` 命令中查到。

## 模型预设文件

预设文件存放在 `models/*.sh`，每个文件对应一个模型配置。新增模型时复制 `models/template.sh` 修改即可。

**可用预设：**

| 文件 | 模型 | TP | 说明 |
|------|------|----|------|
| `qwen3-32b.sh` | Qwen3-32B | 2 | 正式打分用，需双卡 |
| `qwen3-4b.sh` | Qwen3-4B | 1 | 单卡，基础版 |
| `qwen3-4b-judge.sh` | Qwen3-4B (微调版) | 1 | 单卡，judge 微调 |
| `qwen3-4b-thinking.sh` | Qwen3-4B (thinking) | 1 | 单卡，thinking 模式 |
| `template.sh` | — | — | 新预设的起点 |

**预设文件中的关键变量：**

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL` | HF 路径或 repo id | 模型文件位置 |
| `MODEL_NAME` | 字符串 | 服务对外暴露的模型名 |
| `PORT` | `31877` | HTTP 端口，judge 配置需与此一致 |
| `TP` | `1` | tensor parallel，即占用的 GPU 数量 |
| `DP` | `1` | data parallel |
| `TOOL_CALL_PARSER` | `qwen` | Qwen 系列模型填 `qwen`，其余留空 |
| `CONTEXT_LENGTH` | — | 最大上下文长度，不填则用模型默认 |

所有变量均可通过环境变量覆盖，也可以通过命令行选项覆盖（命令行优先级最高）。

## 运行时目录

服务启动后会在项目根目录下生成以下目录：

```text
logs/serve/          # SGLang 服务日志和 PID 文件
cache/serve/         # 运行时缓存（FlashInfer、Triton、HuggingFace 等）
```

`cache/` 目录安全可删，删除后下次启动会重新生成（首次启动会慢一些）。

脚本会自动将以下缓存变量重定向到项目目录，以兼容 `/root` 只读的容器环境：

- `FLASHINFER_WORKSPACE_DIR`
- `TRITON_CACHE_DIR`
- `TORCHINDUCTOR_CACHE_DIR`
- `CUDA_CACHE_PATH`
- `NUMBA_CACHE_DIR`
- `HF_HOME`
- `XDG_CACHE_HOME`

如需自定义路径，设置环境变量 `LOG_DIR=/path` 或 `CACHE_DIR=/path` 即可。

## 与 judge runner 配合

服务启动后，在另一个终端运行 judge：

```bash
cd data_quality/judge

# 使用 SGLang 本地服务（judge_all_metrics.yaml）
python run_judge.py --config configs/judge_all_metrics.yaml

# 使用 OpenAI 兼容 API（judge_api_all_metrics.yaml）
MODEL=qwen3-32b BASE_URL=http://... API_KEY=xxx \
python run_judge.py --config configs/judge_api_all_metrics.yaml
```

输出结果在：

```text
judge/outputs/<run_name>/<YYYYMMDD_HHMMSS>/
  raw_responses.jsonl   # LLM 原始输出
  task_results.jsonl    # 解析后的打分结果
  summary.json          # 各指标汇总统计
  run_config.yaml       # 本次运行的配置快照
```

## 常见问题

**Q: 服务起来了但 judge 连接失败？**

检查 judge 配置文件中的 `client.host` 和 `client.port` 是否与预设文件一致（默认 `127.0.0.1:31877`）。SLURM 场景下 host 需要改为节点实际 IP。

**Q: 启动报错 "CUDA out of memory"？**

减小 `MEM_FRACTION_STATIC` 或降低 `MAX_RUNNING_REQUESTS`，也可以换用参数量更小的预设（如 `qwen3-4b.sh`）。

**Q: 想换模型但不想改预设文件？**

用命令行覆盖：`--model /new/path --tp 1`，不会修改预设文件。

**Q: 同时跑多个服务实例？**

修改 `--port` 避免端口冲突，同时在 judge 配置里对应修改 `client.port`。
