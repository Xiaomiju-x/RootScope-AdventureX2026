# RootScope seed17 BPU：X5 手工隔离只读运行说明

当前口径：`MANUAL_ISOLATED_SUPPORT_ONLY_NOT_X5_OR_MODEL_QUALIFIED`。

这条链路只用于在一台已经人工确认身份的 RDK X5 上，手工验证一个**由发布清单给出 SHA-256** 的 seed17 Bayes-e `.bin`。它不会安装或复制模型，不进入 RootScope 常驻服务，不连接串口、状态机或水泵，也不会产生灌溉执行权限。

配套机器合同是 `seed17_bpu_isolated_runtime_contract.json`，入口是 `scripts/bpu_seed17_isolated_readonly.py`。即使一次真机推理成功，以下状态仍固定为 `false`：

- `x5_ready`、`x5_validated`；
- `camera_qualified`；
- `model_candidate`、`model_qualified`；
- `production_integration_allowed`、`production_authority_enabled`；
- `irrigation_authority_enabled`、`execution_authority`、`physical_authority`。

## 0. 必须先建独立 BPU venv

**不能使用 core v1 的默认 venv 跑 BPU。** core v1 venv 没有 `--system-site-packages`，看不到 RDK OS 自带的 `hobot_dnn`；如果再把 core wheelhouse 里的 NumPy 2.2 装进去，还有覆盖板端 NumPy/`hobot_dnn` ABI 的风险。

BPU 只允许使用独立的 CPython 3.10 `--system-site-packages` venv：NumPy 和 `hobot_dnn` 必须继续来自 RDK 系统路径，不能位于这个 venv 内。准备脚本只做建 venv 和 import-only 检查，不加载 `.bin`、不 forward、不打开设备：

```bash
cd /opt/rootscope/current

/usr/bin/python3 deploy/x5/scripts/prepare_bpu_system_site_venv.py \
  --venv "$HOME/.venvs/rootscope-bpu" \
  --output-json "$HOME/rootscope_evidence/seed17_bpu_venv.json"

BPU_PY="$HOME/.venvs/rootscope-bpu/bin/python"
```

若 RDK 系统 Python 已有 Pillow，不需要安装任何 wheel。若 import-only 回执明确显示 Pillow 缺失，只能使用发布清单中哈希锁定的本地 aarch64/CPython 3.10 Pillow wheel：

```bash
PILLOW_WHEEL=/opt/rootscope/wheelhouse/pillow-12.2.0-cp310-cp310-manylinux2014_aarch64.manylinux_2_17_aarch64.whl

/usr/bin/python3 deploy/x5/scripts/prepare_bpu_system_site_venv.py \
  --venv "$HOME/.venvs/rootscope-bpu" \
  --pillow-wheel "$PILLOW_WHEEL" \
  --expected-pillow-sha256 '<release-manifest 中的 Pillow SHA-256>' \
  --output-json "$HOME/rootscope_evidence/seed17_bpu_venv.json"
```

该脚本强制 `pip --no-index --no-deps`，本地安装白名单只有 Pillow；不会安装 NumPy 或 `hobot_dnn`。BPU CLI 还会二次检查 `pyvenv.cfg` 的 `include-system-site-packages=true`，并拒绝 NumPy/`hobot_dnn` 来源落在 venv 内的环境。

## 1. 冻结接口

运行输入不是 NV12/pyramid，而是后续 release manifest 所绑定候选的冻结 `RGB + NCHW + DDR` 合同：

1. 输入为 OpenCV/UVC 的 `uint8 BGR HxWx3`；
2. 短边按 PIL bilinear 缩放到 256；
3. 中心裁剪 224×224；
4. BGR 转 RGB，再转成连续 `uint8 [1,3,224,224] NCHW`；
5. 主机端**不**做 `/255`、mean 或 std；ImageNet mean/scale 已编进 `.bin`；
6. `pyeasy_dnn` 必须暴露恰好一个模型、一个输入和一个输出；输入元数据必须为 NCHW/`[1,3,224,224]`/uint8-RGB；
7. forward 必须返回数值型、全有限的 `[1,4]` logits，类别顺序固定为 `grass_clump / low_shrub / young_tree / unknown`。

模型哈希、接口、输出 shape 或 finite 任一不符都会失败，不存在 CPU、随机权重或假 BPU 回退。

## 2. 默认预检：不打开相机、不 forward

下面的 `MODEL_SHA` 必须逐字来自冻结 release manifest 或受控交接记录。**不能临场对待测 `.bin` 自算哈希再把该值当作期望值**，否则失去防篡改意义。

```bash
cd /opt/rootscope/current

MODEL_BIN=/opt/rootscope/models/rootscope_seed17_resnet18_224x224_rgb_ddr.bin
MODEL_SHA='<release-manifest 中的 64 位小写 SHA-256>'

"$BPU_PY" deploy/x5/scripts/bpu_seed17_isolated_readonly.py \
  --model-bin "$MODEL_BIN" \
  --expected-model-sha256 "$MODEL_SHA" \
  --output-json "$HOME/rootscope_evidence/seed17_bpu_preflight.json"
```

默认模式只做：Linux/aarch64 门、`.bin` SHA、`hobot_dnn` 导入、模型加载和输入输出接口检查。它不会打开相机，也不会执行 forward。真实 `dnn.load()` 可能分配 BPU/CMA，因此报告会如实记录 `hardware_touched=true`；进程退出后不保留常驻服务。

成功状态仅为：

`HASH_AND_INTERFACE_PREFLIGHT_PASS_NOT_X5_OR_MODEL_QUALIFICATION`

这不是性能、准确率、摄像头路径或整机资格证据。

## 3. 黄金图单次回放

黄金图也必须来自冻结清单，并同时提供文件 SHA。脚本先验图像文件哈希，再解码和执行一次 forward：

```bash
IMAGE=/opt/rootscope/golden/grass_reference_01.png
IMAGE_SHA='<黄金图清单中的 64 位小写 SHA-256>'

"$BPU_PY" deploy/x5/scripts/bpu_seed17_isolated_readonly.py \
  --model-bin "$MODEL_BIN" \
  --expected-model-sha256 "$MODEL_SHA" \
  --image "$IMAGE" \
  --expected-image-sha256 "$IMAGE_SHA" \
  --output-json "$HOME/rootscope_evidence/seed17_bpu_golden_replay.json"
```

回执记录模型/图像/预处理 tensor 哈希、原始 logits、softmax、raw top1、模型元数据以及全套 authority 边界。raw top1 只能称为 `RAW_TOP1_HYPOTHESIS_NOT_OPEN_WORLD_ACCURACY_EVIDENCE`，不能写成开放世界沙漠植物识别已资格化。

## 4. 可选 UVC 单帧：必须显式设备路径

只有人工确认摄像头接线和设备别名后，才可显式选择一个 `/dev/...` 字符设备。脚本不扫描 `/dev`，不接受数字相机索引，只打开这一条路径、读取一帧、立即释放：

```bash
"$BPU_PY" deploy/x5/scripts/bpu_seed17_isolated_readonly.py \
  --model-bin "$MODEL_BIN" \
  --expected-model-sha256 "$MODEL_SHA" \
  --camera-device /dev/rootscope_uvc \
  --output-json "$HOME/rootscope_evidence/seed17_bpu_uvc_one_frame.json"
```

即使成功，状态仍是：

`ISOLATED_ONE_FRAME_BPU_REPLAY_PASS_NOT_CAMERA_QUALIFICATION`

单帧成功不证明分辨率、曝光、持续帧率、热稳定性、USB 拔插恢复或打印卡现场域已经通过。

## 5. 回执判读和失败处理

- `runtime.injected_test_backend` 在真机必须为 `false`；fake 后端只存在于 PC 单元测试，状态永远带 `NOT_BPU_EVIDENCE`。
- 真机 forward 后，`runtime.backend` 应为 `hobot_dnn.pyeasy_dnn`、`runtime.forward_executed=true`、`authority.bpu_used=true`。
- `authority.serial_write/state_machine_write/pump_command/irrigation_execution/execution_authority/physical_authority` 必须全为 `false`。
- 任何失败统一返回非零退出码和 `FAIL_CLOSED_NO_AUTHORITY`。如果已尝试加载 BPU 或打开相机，失败回执保守标记 `hardware_touched=true`，但仍不授予任何 authority。
- 不要把这个 CLI 放进 systemd，不要自动重试，不要循环摄像头，不要与生产 RootScope 服务并发争用相机或 CMA。

## 6. PC 测试证据边界

`tests/test_seed17_bpu_isolated_runtime.py` 使用显式 fake dnn，只验证通道、几何、NCHW 连续布局、模型/图像哈希、输入输出 shape、finite、异常路径和零权限回执。它不导入真实 `hobot_dnn`，不接触 BPU、相机、网络或执行器，不能替代 X5 现场回放。

```bash
cd /opt/rootscope/current
python3 -m unittest tests.test_seed17_bpu_isolated_runtime -v
```

只有后续独立的 X5 哈希回执、黄金图回放、摄像头专项资格和整机安全验收全部通过，才能在另一个受控 release 中讨论提升任何资格字段；本合同本身永远保持全 false。
