# RootScope 全新 RDK X5 离线部署手册

## 先看结论

`rootscope_x5_offline_core_v1.tar` 是当前可交付的 **CPU 实验模型离线胶囊**：包含 RootScope 只读代码、seed17 ONNX、CPython 3.10/aarch64 候选 wheelhouse、OpenCV 双路径代码和哈希合同。它不包含数据集，不启动服务，不打开相机、串口或执行器，也不授予灌溉权限。

当前正式状态始终是：`NOT_EXACT_TWIN_X5_QUALIFIED`、`model_candidate=false`、`model_qualified=false`、`x5_validated=false`、`execution_authority=false`。

另外有一个独立大文件包 `rootscope_x5_readonly_llm_model_v1.tar`，只包含只读讲解用的 Qwen2 0.5B GGUF 及其 manifest。它 **没有** 适配 RDK OS 的已资格化 `llama-server`，因此不能把“模型已打包”讲成“LLM 已可运行”。

## 0. 全新板卡的离线前提

- RDK X5 已安装官方系统，当前 shell 是普通用户；不需要创建 `rootscope` 用户，也不使用 `sudo`。
- [地瓜机器人官方 Model Zoo 概述](https://developer.d-robotics.cc/model_zoo_doc/model_zoo_intro)要求 RDK X5 主交付分支使用 RDK OS `>= 3.5.0`；RDK X5 系统基于 Ubuntu 22.04 aarch64，并使用 TROS Humble 生态。现场应人工记录镜像版本；本 v1 安装脚本只机械校验 Linux/aarch64/CPython 3.10，**不会读取或声称已验证 RDK OS 版本**。
- 系统必须是 `Linux aarch64`，解释器必须是 `CPython 3.10`。
- 系统 Python 必须能执行 `python3 -m venv`。安装包不会联网补依赖；若系统镜像缺少 venv/ensurepip，脚本会 fail-closed，需要先用赛前冻结的系统镜像或离线系统包补齐。
- 把两个 `.tar` 及各自 `.sha256` 文件用 U 盘或用户选择的文件传输方式放到 X5。本手册不修改 SSH、Wi-Fi、VPN、代理、路由或网卡。

## 1. 解包前校验

```bash
cd /path/to/transfer
sha256sum -c rootscope_x5_offline_core_v1.tar.sha256
tar -tf rootscope_x5_offline_core_v1.tar | less
tar -xf rootscope_x5_offline_core_v1.tar
```

归档根目录固定为 `rootscope_x5_offline_core_v1/`。不要跳过外层 `.sha256`；安装脚本还会再次逐文件校验内部 `release_manifest.json` 与 `SHA256SUMS`。

## 2. 一键离线安装并做 CPU 自检

```bash
cd rootscope_x5_offline_core_v1
bash ./install_and_selftest.sh
```

默认安装到当前用户：

```text
$HOME/.local/share/rootscope/releases/rootscope_x5_offline_core_v1/rootscope
$HOME/.local/share/rootscope/venvs/rootscope_x5_offline_core_v1
$HOME/.local/share/rootscope/config/rootscope_x5_offline_core_v1.capsule.json
$HOME/.local/share/rootscope/evidence/rootscope_x5_offline_core_v1/
```

需要换盘时只能显式指定一个当前用户可写目录：

```bash
bash ./install_and_selftest.sh --install-base /absolute/user/writable/path
```

脚本顺序为：平台门 → 包内全量哈希 → 复制只读 payload → 候选 wheelhouse 审计 → `pip --no-index --require-hashes` 建 venv → CPU/OpenCV import → capsule preflight → golden preprocessing → ONNX 模拟 RGB 回放。它不会调用 `systemctl`，不会启动 Dashboard/LLM，不会枚举 `/dev`，不会访问网络。

看到 `PASS_LOCAL_AARCH64_CPU_SMOKE_NOT_X5_QUALIFIED` 只证明这一次普通 aarch64 Linux 本机 CPU 冒烟通过；它不是精度证据，也不会自动把任何 `qualified` 或 `x5_ready` 字段改成 true。

## 3. 手工复跑只读 CPU 证据

```bash
BASE="$HOME/.local/share/rootscope"
PROJECT="$BASE/releases/rootscope_x5_offline_core_v1/rootscope"
PYTHON="$BASE/venvs/rootscope_x5_offline_core_v1/bin/python3"
CONFIG="$BASE/config/rootscope_x5_offline_core_v1.capsule.json"

cd "$PROJECT"
"$PYTHON" -m app.edge.cli preflight --config "$CONFIG"
"$PYTHON" -m app.edge.cli selftest --config "$CONFIG"
```

保留 `evidence/rootscope_x5_offline_core_v1/` 中的 JSON。不要只拍终端成功画面；模型 SHA、golden tensor SHA、provider、输出 shape 与所有 false 权限字段应一起保存。

## 4. 可选 OpenCV／已登记打印卡双路径

安装流程只 import OpenCV，不打开相机。core 已带冻结注册表 `app/vision/known_card_template_registry.frozen.experimental.json`（SHA-256 `f5328b66c6b385dd081c1a27577d92b5c63fc18dc00d815f7d3d2561bb68e29f`）及三张模板：`163498042` 草丛、`68787114` 灌木、`92774234` 幼树。它们是训练集中的 `DEMO_REFERENCE_NOT_HOLDOUT_ONCE_REGISTERED`，只用于已知打印卡演示，绝不是 holdout 或泛化证据。`157364276` unknown/沙丘页保持未登记，只作为拒绝负例。打印卡 PDF 的冻结 SHA-256 为 `dfd4b2e9524f2a37fbe39b9f1911b441c0b44565da93e4cfd321c2afe248070a`，PDF 不放入运行 core，需与打印物料单独携带。

```bash
cd "$PROJECT"
"$PYTHON" -m app.vision.dual_path_demo \
  --query /path/to/existing_capture.png \
  --registry app/vision/known_card_template_registry.frozen.experimental.json \
  --capsule-config "$CONFIG" \
  --thresholds-json app/vision/dual_path_demo.thresholds.example.json \
  --matcher-config-json app/vision/card_geometric_matcher.config.example.json \
  --output-json /path/to/evidence/capture.dual_path.json
```

退出码 `0` 也只表示 `EXPERIMENTAL_KNOWN_CARD_CONSENSUS`；它不是开放世界植物识别，不是 holdout 精度，也不能触发泵、STM32 或状态机。空注册表位于 `app/vision/known_card_template_registry.empty.example.json`，空表运行应被拒绝。

PC fixture 软件烟测已绑定在 `evidence/rootscope_dual_path_pc_fixture_audit_20260717.json`：三张 positive 模拟复拍均产生共识，unknown 负例被拒绝，40/40 检查通过。它只证明冻结软件与 fixture 合同，**不是** X5、真实 UVC、现场光照或泛化精度证据。

若必须从 UVC 相机现场采一帧，使用显式设备别名的一次性入口；禁止传相机序号，禁止自动枚举：

```bash
"$PYTHON" deploy/x5/scripts/capture_and_dual_path_once.py \
  --device /dev/rootscope_uvc \
  --capture-png /path/to/evidence/capture_001.png \
  --capture-receipt /path/to/evidence/capture_001.capture.json \
  --dual-path-output /path/to/evidence/capture_001.dual_path.json \
  --registry app/vision/known_card_template_registry.frozen.experimental.json \
  --capsule-config "$CONFIG" \
  --thresholds-json app/vision/dual_path_demo.thresholds.example.json \
  --matcher-config-json app/vision/card_geometric_matcher.config.example.json \
  --width 1280 --height 720 --warmup-frames 2
```

该命令只打开用户指定且解析后仍位于 `/dev` 的一个字符设备，采图后立即释放，再把 PNG 交给只读双路径 CLI。capture receipt 会如实记录 `camera_opened=true`，但泵、串口、状态机、服务启动与全部执行权限仍是 false；安装脚本和 CPU 自检绝不会自动调用它。

## 5. 可选本地 LLM：默认停用

先校验并解开独立模型包：

```bash
cd /path/to/transfer
sha256sum -c rootscope_x5_readonly_llm_model_v1.tar.sha256
tar -xf rootscope_x5_readonly_llm_model_v1.tar
```

GGUF 已锁定，但 `llama-server` 不在包内。只有拿到 **针对现场 RDK OS / aarch64 构建并单独锁定 SHA-256** 的可执行文件后，才能运行 `deploy/x5/scripts/install_readonly_llm.py` 做“安装但不启用”。不要直接使用在 Ubuntu 24.04 构建、而现场系统兼容性未验证的通用二进制。

LLM 用户 unit 无硬编码账号、无 `[Install]` 自动启用、默认无 gate/ack，只允许 loopback；但在 `llama-server` 真板资格化前，演示应使用确定性模板讲解。LLM 输出永远只能读结构化快照并解释，不得有工具调用、串口、泵或执行器权限。

## 6. BPU 当前明确阻塞

- 本 core 只有 CPU ONNX，归档中没有 `.bin`。
- Bayes-e 校准与 mapper 配置已准备，但尚未运行 Horizon mapper，`bpu_compiled=false`。
- 未取得用户对 Docker/WSL 网络接口影响的明确授权前，不能启动 Docker/WSL 做编译。
- 后续即使得到 `.bin`，也必须在 X5 上完成模型加载、输入合同、输出顺序、CPU/BPU 漂移和持续延迟回放，才能单独讨论 BPU 资格；不得修改本 v1 的 false 历史记录。

## 7. 现场最小演示顺序

1. 展示外层 archive SHA 与包内 manifest 校验。
2. 展示 `preflight`：CPU provider、模型 SHA、无设备打开、权限全 false。
3. 展示 `selftest`：golden preprocessing 与 `[1,4]` ONNX 输出。
4. 再演示一张已登记植物卡和一张沙丘负例，保留两份 JSON。
5. 最后讲解 LLM/BPU 的真实状态：可选模型已分包；server/BPU bin 尚未资格化，系统有确定性降级路径。

这套口径的亮点是可复现、可解释、可拒绝和可审计，而不是把未完成的硬件能力写成已经完成。
