# AIRE-DIAL

AIRE-DIAL（Distributed Inference and Layering）是一个面向边缘设备和异构硬件的分布式大模型推理系统。它用 Rust 实现，将模型 Transformer 层按拓扑配置分配到 Master 和多个 Worker，通过持久 TCP 连接传输张量，在一个统一的 HTTP API 上提供文本、图像和视频问答能力。

当前代码主要覆盖以下场景：

- Qwen3-VL-2B/8B、Llama3 等模型的原生分层推理；
- RK3588 CPU/NPU、Jetson Orin CUDA/GGUF 等异构节点协同；
- Qwen3-VL 的文本、图片和视频输入；
- OpenAI Chat Completions 风格 API、SSE 流式输出和内置 Web Chat；
- 静态拓扑和基于性能画像的自动层规划；

> 本 README 说明当前源码的通用部署方式。特定硬件优化、模型转换和论文实验请先阅读文末的专项文档。

Qwen3.8-27B 的 Thor + Orin 量化分层路径新增 `qwen38-ggml`：
[构建、GGUF 转换与启动说明](docs/qwen38_ggml.md)。它保留 DIAL 的层拓扑和张量协议，
本地层调用上游 GGML 算子；不是 `qwen38-rpc` 的服务器代理模式。

## 1. 系统架构

```text
                         HTTP / Web Chat / CLI
                                  |
                                  v
                    +---------------------------+
                    | Master                    |
                    | API、会话、采样、调度      |
                    | 本地层、视觉、KV cache     |
                    +-------------+-------------+
                                  |
                 DIAL binary tensor protocol (TCP)
                    +-------------+-------------+
                    |             |             |
                    v             v             v
              +-----------+ +-----------+ +-----------+
              | Worker 0  | | Worker 1  | | Worker 2  |
              | model     | | model     | | model     |
              | layers    | | layers    | | layers    |
              +-----------+ +-----------+ +-----------+
```

### 角色

- **Master**：启动 HTTP 服务，读取输入和会话历史，执行本地层、视觉编码、KV cache、采样，并把远程层请求发送给 Worker。
- **Worker**：读取同一模型目录中自己负责的层，监听 TCP 端口，执行层前向计算并返回张量。Worker 不提供 HTTP API。
- **Client**：可以是 `dial-cli --api-client`、`curl`、浏览器 Web Chat 或其他 OpenAI 风格客户端。

Master 和 Worker 必须使用兼容的代码版本、模型基座、tokenizer 和 dtype。拓扑文件决定哪一个 Worker 负责哪些层；没有被拓扑分配的层留在 Master 本地执行。

## 2. 目录结构

```text
Dial_llama/
├── dial-core/                 # 模型、Master/Worker、协议、HTTP API
├── dial-cli/                  # 命令行入口和 API 客户端
├── vendor/rknpu2/             # RKNN Rust 绑定
├── tools/                     # 视频、评测、模型转换和规划工具
├── docs/                      # 专项部署与实验文档
├── topology*.yml              # 静态拓扑示例
├── transmodel/                # RKNN/RKLLM 等转换产物（按需准备）
├── web_chat/                  # 内置 Web Chat 页面
├── Cargo.toml                 # Cargo workspace
└── Makefile                   # 常用构建命令
```

## 3. 环境要求

### 3.1 通用环境

- Linux x86_64 或 Linux aarch64；RK3588 通常使用 aarch64，Jetson Orin 使用 aarch64；
- Rust stable、Cargo 和可用的 C/C++ linker；
- Git、`pkg-config`、`cmake`、`make`；
- 至少一台设备可以完整读取模型文件，所有参与推理的 Master/Worker 都应能访问对应模型目录；
- Master 与 Worker 之间可以互相访问 TCP 端口，默认 Worker 端口是 `10128`，Master HTTP 端口示例是 `8082`。

Ubuntu/Debian 可以先安装基础工具：

```bash
sudo apt update
sudo apt install -y \
  build-essential pkg-config cmake make git curl \
  ffmpeg netcat-openbsd htop
```

安装 Rust（若系统尚未安装）：

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
rustc --version
cargo --version
```

当前开发服务器已检查到 `x86_64`、Rust `1.92.0`、Cargo `1.92.0`、系统 C 编译器和 `pkg-config`。视频工具位于 `/opt/sophon/sophon-ffmpeg-latest/bin/`；如果该目录不在 `PATH`，启动 Master 前执行：

```bash
export DIAL_FFMPEG_BIN=/opt/sophon/sophon-ffmpeg-latest/bin/ffmpeg
export DIAL_FFPROBE_BIN=/opt/sophon/sophon-ffmpeg-latest/bin/ffprobe
```

### 3.2 模型目录

原生 DIAL 后端从 `--model` 指定的目录读取 Hugging Face 风格模型。至少应检查：

```bash
MODEL=/path/to/Qwen3-VL-8B-Instruct
test -f "$MODEL/config.json"
test -f "$MODEL/model.safetensors.index.json"
test -f "$MODEL/tokenizer.json" || test -f "$MODEL/tokenizer.model"
```

`config.json` 中的 `model_type` 用于自动选择实现：

| `model_type` | DIAL 实现 |
| --- | --- |
| `qwen3_vl` | Qwen3-VL 文本/视觉原生实现 |
| 其他或缺省 | Llama3 兼容实现 |

不要混用 Instruct 和 Thinking 模型目录、不同 tokenizer，或不同版本的模型权重。

### 3.3 CPU、CUDA 和 RKNN

- **CPU**：直接使用默认构建，不需要 CUDA；启动时加 `--cpu`。
- **CUDA**：需要 NVIDIA 驱动、CUDA Toolkit 和与 Candle 兼容的编译环境，使用 `cargo build --release --features cuda`；Worker 可选 W8A16 或 GGUF。
- **RKNN**：仅在目标板的 Linux/aarch64 环境使用，除 DIAL 外还需要 `librknnrt.so` 和已经转换好的 `.rknn` 文件。可以通过 `--vision-rknn-lib`、`--text-rknn-lib` 等参数显式指定 runtime。

## 4. 编译

```bash
cd /path/to/Dial_llama

# 默认构建：包含 Master API，适合 CPU 或 RKNN 设备
cargo build --release

# CUDA 构建：在需要 Candle CUDA 的设备上执行
cargo build --release --features cuda

# 运行单元测试
cargo test

# 查看完整命令行参数
./target/release/dial-cli --help
```

Makefile 等价命令：

```bash
make build_release
make test
```

成功后主要产物是 `target/release/dial-cli`。如果需要把构建产物放到其他磁盘，使用 `CARGO_TARGET_DIR=/path/to/target cargo build --release`。不要在一台设备上复用另一台设备的 CUDA、RKNN 或系统架构构建产物。

## 5. 最小单机启动

单机验证时使用仓库自带的空拓扑，避免默认拓扑指向不存在的远程 Worker：

```bash
cd /path/to/Dial_llama

RUST_LOG=info ./target/release/dial-cli \
  --mode master \
  --api 0.0.0.0:8082 \
  --model /path/to/Qwen3-VL-8B-Instruct \
  --topology docs/topology_solo.example.yml \
  --cpu \
  --sample-len 128
```

看到 `starting api on http://0.0.0.0:8082 ...` 后，Master API 已监听。

在当前服务器 `/home/seaway/sdb/ljl` 上，可以直接使用已经存在的模型目录执行：

```bash
cd /home/seaway/sdb/ljl/Dial_llama
cargo build --release

RUST_LOG=info ./target/release/dial-cli \
  --mode master \
  --api 0.0.0.0:8082 \
  --model /home/seaway/sdb/ljl/model/Qwen3-VL-8B-Instruct \
  --topology /home/seaway/sdb/ljl/Dial_llama/docs/topology_solo.example.yml \
  --cpu \
  --text-decode-mode cpu-only \
  --sample-len 128
```

### 5.1 使用内置命令行客户端

另开终端执行：

```bash
./target/release/dial-cli \
  --api-client http://127.0.0.1:8082 \
  --ask '1+1 等于多少？' \
  --stream \
  --metrics
```

### 5.2 使用浏览器 Web Chat

打开 `http://127.0.0.1:8082/`。如果 Master 运行在远程设备，将 `127.0.0.1` 换成 Master 的实际 IP，并确保防火墙允许访问 `8082`。

### 5.3 使用 HTTP API

```bash
curl http://127.0.0.1:8082/api/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "请介绍一下 DIAL。"}],
    "stream": false
  }'
```

流式请求：

```bash
curl -N http://127.0.0.1:8082/api/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "用三句话介绍 DIAL。"}],
    "stream": true
  }'
```

## 6. 多机 Master/Worker 部署

以下示例使用一台 Master 和两台 Worker：

| 角色 | IP | 端口 |
| --- | --- | --- |
| Master | `192.168.1.80` | HTTP `8082` |
| Worker 0 | `192.168.1.81` | TCP `10128` |
| Worker 1 | `192.168.1.82` | TCP `10128` |

### 6.1 编写拓扑文件

Qwen3-VL-8B 有 36 个语言层，使用 `model.language_model.layers.N` 作为层名前缀：

```yaml
# topology_qwen3vl_2workers.yml
worker0:
  host: "192.168.1.81:10128"
  description: "Qwen3-VL worker 0"
  layers:
    - "model.language_model.layers.0-17"

worker1:
  host: "192.168.1.82:10128"
  description: "Qwen3-VL worker 1"
  layers:
    - "model.language_model.layers.18-35"
```

范围表达式会在启动时展开。拓扑中的 Worker 名称必须与启动参数 `--name` 对应。没有写入拓扑的 embedding、视觉编码、final norm、lm_head 以及其他层由 Master 本地执行。

仓库内已有可参考的文件：

- [topology_qwen3vl.yml](topology_qwen3vl.yml)
- [docs/topology_solo.example.yml](docs/topology_solo.example.yml)
- [docs/topology_even_4devices.example.yml](docs/topology_even_4devices.example.yml)
- [topology_qwen3vl_worker_gguf.yml](topology_qwen3vl_worker_gguf.yml)

### 6.2 启动 Worker

在 `192.168.1.81`：

```bash
cd /path/to/Dial_llama
RUST_LOG=info ./target/release/dial-cli \
  --mode worker \
  --name worker0 \
  --address 0.0.0.0:10128 \
  --model /path/to/Qwen3-VL-8B-Instruct \
  --topology /path/to/topology_qwen3vl_2workers.yml \
  --cpu
```

在 `192.168.1.82` 使用相同命令，但将 `--name worker0` 改为 `--name worker1`。如果 Worker 使用 CUDA，则去掉 `--cpu`，并确保二进制是用 `--features cuda` 构建的。

### 6.3 检查连通性

在 Master 上执行：

```bash
nc -vz 192.168.1.81 10128
nc -vz 192.168.1.82 10128
```

若系统没有 `nc`，可安装 `netcat-openbsd`，或者使用 `timeout 2 bash -c '</dev/tcp/192.168.1.81/10128'` 检查端口。

### 6.4 启动 Master

在 `192.168.1.80`：

```bash
cd /path/to/Dial_llama
RUST_LOG=info ./target/release/dial-cli \
  --mode master \
  --api 0.0.0.0:8082 \
  --model /path/to/Qwen3-VL-8B-Instruct \
  --topology /path/to/topology_qwen3vl_2workers.yml \
  --cpu \
  --sample-len 128
```

然后在任意能访问 Master 的机器上提问：

```bash
./target/release/dial-cli \
  --api-client http://192.168.1.80:8082 \
  --ask '请说明当前请求经过了哪些节点。' \
  --metrics
```

所有节点的 `--model`、拓扑文件和模型基座必须匹配。Worker 只加载自己负责的层，但仍需要能够读取模型索引和对应权重分片。

## 7. 文本、图片和视频推理

### 7.1 文本

```bash
./target/release/dial-cli \
  --api-client http://127.0.0.1:8082 \
  --ask '用一句话解释 KV cache。'
```

### 7.2 图片

`--image` 会读取本地图片并编码为 base64，再发送为 OpenAI 风格的多模态消息：

```bash
./target/release/dial-cli \
  --api-client http://127.0.0.1:8082 \
  --image ./test.png \
  --ask '请描述图片中的主要内容。' \
  --stream \
  --metrics
```

Qwen3-VL 的图片尺寸可以限制为最大边长，或者固定为硬件模型需要的尺寸：

```bash
# 软件视觉路径：只缩小大图，不放大小图
./target/release/dial-cli \
  --mode master --api 0.0.0.0:8082 \
  --model /path/to/Qwen3-VL-8B-Instruct \
  --topology /path/to/topology_solo.yml \
  --cpu --vision-max-side 384 --vision-no-upscale

# RKNN 静态输入路径示例
./target/release/dial-cli \
  --mode master --api 0.0.0.0:8082 \
  --model /path/to/Qwen3-VL-8B-Instruct \
  --topology /path/to/topology_solo.yml \
  --cpu \
  --vision-rknn /path/to/vision_448.rknn \
  --vision-fixed-side 448
```

### 7.3 视频

默认按 Qwen3-VL 规则在内存中采样视频，默认 2 FPS、最少 4 帧、最多 768 帧：

```bash
./target/release/dial-cli \
  --api-client http://127.0.0.1:8082 \
  --video ./demo.mp4 \
  --ask '请分析视频中的事件发展，并指出关键时间点。' \
  --metrics
```

参数说明：

- `--video-max-bytes`：原始视频大小上限，默认 `268435456`（256 MiB），超限直接拒绝；
- `--video-fps`：采样帧率，默认 `2`，设置为 `0` 时按最大帧数均匀采样；
- `--video-min-frames` / `--video-max-frames`：采样帧数上下限；
- `--video-max-side`：采样帧的空间边长上限，默认 `128`；
- `--video-batch-size`：视觉时间 patch 批大小，增大可以提高吞吐但会增加内存；
- `--video-no-sample`：处理全部原始帧，只适合短视频或对照实验。

也可以使用独立视频客户端或无限视频流脚本：

```bash
python3 tools/video_inference_client.py \
  --api-client http://127.0.0.1:8082 \
  --video ./demo.mp4 \
  --ask '视频里发生了什么？'

python3 tools/stream_video_client.py --help
```

视频推理需要系统可以找到 `ffmpeg` 和 `ffprobe`。如果不在 `PATH`，设置 `DIAL_FFMPEG_BIN` 和 `DIAL_FFPROBE_BIN`。

### 7.4 交互式 REPL

```bash
./target/release/dial-cli \
  --api-client http://127.0.0.1:8082 \
  --repl --stream --metrics
```

REPL 中输入普通文本进行对话；输入 `/img 图片路径 问题` 发送图片，输入 `/video 视频路径 问题` 发送视频，输入 `/quit` 退出。

## 8. Qwen3-VL 的 RKNN/CUDA 优化

这些选项只适用于 Qwen3-VL 后端。

| 参数 | 用途 |
| --- | --- |
| `--vision-rknn` | 用 RKNN 运行 ViT + merger 视觉编码器 |
| `--text-rknn-dir` | 从按层切分的 RKNN chunk 运行文本路径 |
| `--text-rknn-prefill` | 显式开启文本 prefill RKNN |
| `--text-decode-mode auto` | 按已有 chunk 自动选择软件、全 NPU 或 NPU+CPU |
| `--text-decode-mode npu-cpu` | 前缀层走 NPU，剩余层走 CPU/分布式路径 |
| `--text-decode-mode cpu-only` | decode 全部走 CPU/分布式路径 |
| `--text-qkv-rknn-dir` | 只替换 decode 的 Q/K/V 投影，KV cache 和 attention 仍由宿主处理 |
| `--text-mlp-rknn-dir` | 只替换 decode 的 MLP 子图 |
| `--worker-w8a16 true` | CUDA Worker 使用逐行 INT8 权重 |
| `--worker-quantized-gguf` | CUDA Worker 只从 GGUF 读取自己负责的 Transformer 层 |
| `--worker-gguf-output-head true` | 将最后 norm 和输出投影放到最终层 Worker |
| `--worker-gguf-sample-token true` | 在最终层 Worker 采样，只返回一个 token；必须同时开启 output head |

RKNN 转换产物、Qwen3-VL Worker GGUF 和 W8A16 的详细限制见：

- [docs/worker_gguf.md](docs/worker_gguf.md)
- [docs/worker_w8a16.md](docs/worker_w8a16.md)
- [run.md](run.md)

## 9. 拓扑和自动规划

### 9.1 静态拓扑

拓扑文件是一个 `worker name -> node` 的 YAML 映射：

```yaml
worker0:
  host: "192.168.1.81:10128"
  description: "worker description"
  layers:
    - "model.language_model.layers.0-17"
```

端口和地址由 `host` 决定；`layers` 支持单层名和闭区间范围。静态拓扑不会自动均分，也不会自动探测错误的层前缀。

### 9.2 性能画像自动规划

准备包含设备内存、逐层 prefill/decode 耗时、网络 RTT 和带宽的 profile，然后传给 Master：

```bash
./target/release/dial-cli \
  --mode master \
  --api 0.0.0.0:8082 \
  --model /path/to/Qwen3-VL-8B-Instruct \
  --auto-plan-profile docs/auto_plan_4devices.example.yml \
  --auto-plan-algorithm dial \
  --auto-plan-output result/selected-plan.json
```

可选算法：`dial`、`edgeshard-latency`、`edgeshard-throughput`。自动规划启用后，Master 和每个实际参与的 Worker 都必须使用同一个 profile 和算法；profile 中未选中的 Worker 不应启动。参考：[docs/auto_planner_4devices.md](docs/auto_planner_4devices.md)。

## 10. API 和性能指标

原生 Master 提供：

| 地址 | 方法 | 作用 |
| --- | --- | --- |
| `/`、`/chat`、`/web-chat` | GET | 内置 Web Chat |
| `/api/v1/chat/completions` | POST | OpenAI 风格聊天补全，支持 SSE |
| `/api/v1/topology` | GET | 查看拓扑、节点和 Worker 状态 |

使用 `--metrics` 或查看 JSON 响应可以观察：

- `ttft_s`：首 token 延迟；
- `total_s`：请求总耗时；
- `tps` / `tokens_per_second`：整体生成速率；
- `decode_tps` / `decode_tokens_per_second`：去掉首 token 阶段后的解码速率；
- `dist_overhead_s` / `distributed_overhead_s`：网络、序列化和协议等非计算开销；
- `remote_compute_s`：远程 Worker 计算累计时间；
- `remote_requests`：远程请求次数。

## 11. 常用环境变量

这些变量用于调试或实验，不是最小启动所必需的：

| 变量 | 示例 | 作用 |
| --- | --- | --- |
| `RUST_LOG` | `RUST_LOG=info` | 控制日志级别；调试时可用 `debug` |
| `DIAL_CPU_AFFINITY` | `DIAL_CPU_AFFINITY=4-7` | aarch64 上指定 CPU 亲和性 |
| `DIAL_DISABLE_CPU_AFFINITY` | `=1` | 关闭 aarch64 默认 CPU 亲和性 |
| `RAYON_NUM_THREADS` | `=4` | 限制 CPU 并行线程数 |
| `SPM_COMPACT_BATCH` | `=1` | 开启连续远程层 compact batch |
| `SPM_TRACE_TRANSFER` | `=1` | 输出张量传输跟踪日志 |
| `SPM_TRANSFER_LIMIT_MBPS` | `=100` | 模拟或限制传输速率 |
| `SPM_TRANSFER_LIMIT_KBPS` | `=100000` | 以 KB/s 设置传输限制 |
| `DIAL_TEXT_DECODE_PAST_BUCKETS` | `=128,256,512,1024` | 配置文本 RKNN KV cache 的 past 长度 bucket |
| `DIAL_FFMPEG_BIN` | `=/usr/local/bin/ffmpeg` | 指定 ffmpeg 路径 |
| `DIAL_FFPROBE_BIN` | `=/usr/local/bin/ffprobe` | 指定 ffprobe 路径 |
| `DIAL_VIDEO_TEMP_DIR` | `=/data/dial-video-tmp` | 指定视频临时目录 |
| `DIAL_WORKER_GGUF_WARMUP` | `=0` | 关闭 GGUF Worker 启动预热，仅用于冷启动实验 |

性能对比时应固定模型、拓扑、输入、采样参数、频率和网络条件，并记录完整日志；不要只比较单个吞吐数字。

## 12. 监控和故障排查

### 12.1 资源监控

```bash
# NVIDIA GPU
nvidia-smi

# Jetson
tegrastats

# RK3588 NPU
watch -n 1 cat /sys/kernel/debug/rknpu/load

# CPU 和内存
htop

# 查看当前 API 拓扑
curl http://127.0.0.1:8082/api/v1/topology
```

### 12.2 常见问题

**`can't read .../config.json`**

`--model` 必须指向模型目录，而不是单个权重分片；确认 `config.json` 和 `model.safetensors.index.json` 存在且有读取权限。

**`could not find topology for worker`**

`--name` 必须与拓扑中的 key 完全一致。单机测试应使用空拓扑并且不要启动 Worker。

**Master 连接不上 Worker**

检查 Worker 是否已经启动、`host` 是否写成 Worker 的真实 IP、端口是否开放，并在 Master 上执行 `nc -vz HOST PORT`。不要把 Worker 的监听地址写成只对本机可见的 `127.0.0.1`，除非 Master 和 Worker 在同一台机器。

**启动后 OOM**

依次降低 `--kv-cache-max-len`、`--sample-len`、`--vision-max-side` 或 `--video-max-frames`；关闭不必要的 RKNN/FP16/GGUF 双份权重；确认拓扑没有让单个设备承担超出内存的层数。

**图片/视频首 token 很慢**

降低 `--vision-max-side` 或 `--video-max-side`，对小图使用 `--vision-no-upscale`，视频场景减少 `--video-max-frames`。视觉 token 数通常比文本长度更直接地影响 TTFT。

**RKNN runtime 初始化失败**

确认 `.rknn` 的输入 shape、模型目标平台和板端 runtime 版本匹配；必要时显式传入正确的 `--vision-rknn-lib` 或文本 runtime 路径。原生 CPU 模式可先不传任何 RKNN 参数验证模型和拓扑。

**视频工具找不到 ffmpeg**

执行 `which ffmpeg; which ffprobe`，或设置 `DIAL_FFMPEG_BIN` / `DIAL_FFPROBE_BIN` 为绝对路径。

## 13. 安全注意事项

当前 HTTP API 和 Master/Worker TCP 协议没有内置用户认证或加密。`--api 0.0.0.0:8082` 和 `--address 0.0.0.0:10128` 只应绑定在可信局域网、专用网或 VPN 中；不要直接暴露到公网。生产部署应在反向代理、VPN 或防火墙后增加访问控制。

## 14. 专项文档

- [docs/worker_gguf.md](docs/worker_gguf.md)：Orin Worker GGUF 量化路径；
- [docs/worker_w8a16.md](docs/worker_w8a16.md)：CUDA Worker W8A16；
- [docs/auto_planner_4devices.md](docs/auto_planner_4devices.md)：四设备自动层规划；
- [docs/edgeshard_comparison.md](docs/edgeshard_comparison.md)：EdgeShard 对比实验；
- [docs/multi_request_pipeline.md](docs/multi_request_pipeline.md)：多请求流水线、并发参数与测试方法；
- [run.md](run.md)：当前 RKNN/RKLLM 运行记录；
- [run2.md](run2.md)：多板卡、视频流和系统监控记录。

## 15. 许可证

项目许可证见 [LICENSE](LICENSE)。
