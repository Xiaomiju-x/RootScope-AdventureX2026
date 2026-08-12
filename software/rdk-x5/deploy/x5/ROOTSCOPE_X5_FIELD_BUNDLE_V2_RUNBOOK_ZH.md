# RootScope X5 现场离线组合包 v2

本包是“可核验、可分阶段启用”的现场交接包，不代表 RDK X5 真机验收完成。当前默认推理仍是旧 core v1 的 CPU ONNX；r3-r7 BPU 候选全部未通过冻结的 PC 数值漂移门，因此 BPU support 组件**不含 `.bin`**。llama-server 已完成 ARM64 交叉构建和 QEMU 冒烟，但尚无本机 X5 性能、温度或稳定性证据。

## 1. 包含内容

- 逐字节复用的 `rootscope_x5_offline_core_v1.tar`；
- 逐字节复用的只读 LLM GGUF 包；
- ARM64 `llama-server` b9637 包；
- BPU support-only 包：适配器、显式单图 CLI、独立 venv 准备器、合同、说明和**唯一一个 Pillow ARM64 wheel**；
- A4 打印识别卡 PDF；
- 外层 manifest、SHA256SUMS、安装器和本说明。

BPU 环境必须使用 RDK 系统 CPython 3.10 创建的 `--system-site-packages` 独立 venv。不得使用 core v1 venv；不得向 BPU venv 安装 core wheelhouse 中的 NumPy 2.2、ONNX Runtime 或 OpenCV。系统自带 NumPy 与 `hobot_dnn` 的 ABI 必须保留。

## 2. PC 上先只验包

解开外层无压缩 tar 后执行：

```bash
python3 rootscope_x5_field_bundle_v2/install_field_bundle_v2.py \
  --bundle-root rootscope_x5_field_bundle_v2 \
  --verify-only
```

`--verify-only` 不要求 ARM64，不写安装目录，不解包、不枚举设备、不访问网络、不打开串口或相机。成功状态应为 `PASS_HASHES_AND_SAFE_PATHS_NOT_X5_QUALIFIED`。

## 3. 新装 RDK X5 上离线安装

前提：Linux/aarch64、RDK 系统 CPython 3.10，命令必须从系统 Python 运行；先把整个已解开的目录离线复制到 X5，再执行：

```bash
cd rootscope_x5_field_bundle_v2
./install_and_verify.sh
```

默认动作只有：严格校验；安全解出四个嵌套组件；运行 core v1 离线安装和 CPU 模拟输入 self-test；永久暂存 GGUF 和 llama-server；生成仍为 disabled/manual-ack 的用户 unit；创建独立 BPU system-site venv，并在系统没有 Pillow 时用 `--no-index --no-deps` 安装包内唯一 Pillow wheel。

默认**不会**启动 LLM，不会创建启用 gate，不会执行 `systemctl`，不会加载 BPU `.bin`，不会 forward，不会打开相机，不会枚举 `/dev`，不会写 STM32/串口，也不会触发泵或 RootScope 状态机。

安装聚合回执默认位于：

```text
~/.local/share/rootscope-field-v2/evidence/field_bundle_v2_install_receipt.json
```

## 4. LLM 后续人工启用边界

安装器只调用 core 内的 `install_readonly_llm.py`，生成 loopback-only、disabled、manual-ack-required 配置。竞赛前必须在具体 X5 上另行核验内存、温度、首 token 时间和连续运行；在完成前保持 gate 不存在，禁止把 QEMU 数据写成 X5 指标。

## 5. BPU 后续处理

本版 `selected_bin=null`，所以没有可运行的默认 BPU 模型。`bpu_seed17_isolated_readonly.py` 仍要求调用者显式给出一个 `.bin` 和精确 SHA-256；不要从 r3-r7 目录随意挑一个复制到现场包。未来只有新的预先冻结方案通过全部 PC replay 门、再完成 X5 隔离回放后，才能新建一个版本，不得改写本 v2。

## 6. 现场展示口径

- 可以说：CPU 离线链路可自检；ARM64 llama-server 已交叉构建并通过 QEMU loopback 冒烟；BPU 编译实验有完整失败证据，现场包 fail-closed 不带不合格 bin。
- 不可以说：BPU 模型已部署成功、X5 已验证、相机已资格化、本地大模型已在 X5 达到某速度、系统已获得自动灌溉权限。
