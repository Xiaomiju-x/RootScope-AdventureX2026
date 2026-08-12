# RootScope 打印卡 UVC 有界前台

## 1. 这条链做什么

`app.omega_vision.uvc_card_frontend` 面向现场裁开的四张打印卡。它对每一帧分别显示并留证：

1. `semantic`：seed17 CPU ONNX 的原始语义假设；
2. `omega_ood`：Omega 图像质量、Energy、maximum probability 和 pooled-marginal conformal heuristic 的 `CLASSIFY/ABSTAIN`；
3. `geometry`：冻结注册表中三张正类打印模板的特征与透视几何核验；
4. `final`：只有前三层全部一致时才出现 `DISPLAY_ONLY_REGISTERED_CARD_CONSENSUS`。

任何缺件、低质量、`unknown`、OOD、没有且仅有一个几何命中、类别不一致或权限字段异常，都会得到 `SAFE_REJECT_NO_PHYSICAL_AUTHORITY`。这不是开放世界植物识别，不是准确率证据，也不会决定灌溉。

当前植物 BPU 合同仍是：

```text
plant_bpu_selected_bin = null
plant_bpu_used = false
```

所以本前台只运行 seed17 `CPUExecutionProvider`。不得把通用 ImageNet BPU 辅助结果说成植物识别，更不得把它接入最终共识。

## 2. 运行边界

- 必须由操作者传入一个已经登记的 `/dev/v4l/by-id/...-video-indexN`；程序拒绝 `/dev/video0`，也不会扫描或猜摄像头。
- 必须同时传入该摄像头精确的 USB `VID`、`PID` 和 `serial`。程序沿显式设备的 sysfs 祖先链做开前、开后和释放后三次核验，任一变化都 fail-closed。
- 只支持 `one-shot` 或最多 30 帧的 `bounded` 前台会话；达到上限或异常后在 `finally` 中释放相机。
- 不启动后台进程、HTTP 服务或 systemd；不打开串口、GPIO、STM32 或水泵。
- JSONL 是逐记录 `fsync` 的 SHA-256 链；结束后另写一个原子 `*.final.json`，绑定 JSONL 的字节数、文件 SHA、记录数和链根。
- 输出只允许放到操作者显式给出的、规范化且无 symlink 祖先的 `output-root` 直接子项。X5 上固定目录 fd，以 `O_NOFOLLOW|O_EXCL` 创建，并用同目录 hardlink 原子发布 final/标注帧；并发出现的同名目标也绝不覆盖。
- 可选标注帧仅用于现场查看，固定 `training_eligible=false / accuracy_evidence=false`。
- `release()` 返回并不等于释放成功；只有释放后 `isOpened()==false` 且释放后身份仍与开前一致，final 才记录 `camera_released=true`。

## 3. 先冻结摄像头身份

硬件同学先把相机固定插在最终使用的 X5 USB 口，并把稳定路径、VID、PID、serial 交给算法同学。若只知道已经选定的稳定路径，可以只查询这一个目标：

```bash
CAMERA='/dev/v4l/by-id/usb-Web_Camera_Web_Camera_202604081837-video-index0'
udevadm info --query=property --name="$CAMERA" \
  | sed -n -E '/^(ID_VENDOR_ID|ID_MODEL_ID|ID_SERIAL_SHORT)=/p'
```

本次冻结合同必须逐字得到 `32e6`、`9228`、`202604081837`；下方已直接固定这三个值。若重插后任一值不符，立即停止，不要用 `/dev/video0`、不要扫描其他设备，也不要根据产品名猜 serial。

## 4. 需要的冻结资产

下面六项必须存在且由命令显式传入：

| 资产 | 约束 |
|---|---|
| seed17 capsule | 模型启用但 `model_candidate/model_qualified/bpu_ready=false` |
| seed17 ONNX | SHA-256 必须为 `50ae8d2ec1cec0f3748efa127c6b9c00624684f8c112c8a7913c2a0e304a3bad` |
| Omega calibration manifest | 复用 `configs/omega/vision_board_replay_new_x5_20260723.json`；formal coverage 仍为 false |
| 三正类注册表 | `known_card_template_registry.frozen.experimental.json`；unknown 不得登记 |
| 四格打印 manifest | `RootScope_A4_four_up_field_cards_20260723_manifest.json`；必须正好四张卡并保留原角色 |
| demo thresholds / matcher config | v2 冻结 JSON |

这份前台由单独审计的 immutable event-vision overlay v1.2 承载，不覆盖既有 Omega v3 candidate。只有 v1.2 通过外层归档、清单、逐文件哈希、板端预检和测试后，才可称为已经部署到 X5。

## 5. 前台命令

以下只是变量模板。`APP_ROOT` 必须指向包含本模块和 Omega 配置的新只读部署根；`V2_ROOT` 使用现有 v2 安装。

```bash
OVERLAY_ROOT="$HOME/.local/share/rootscope-event-vision-overlay/releases/rootscope_event_vision_overlay_v1_2"
APP_ROOT="$OVERLAY_ROOT/rootscope"
V2_ROOT="$HOME/.local/share/rootscope-field-v2"
CORE="$V2_ROOT/core_v1/releases/rootscope_x5_offline_core_v1/rootscope"
PY="$V2_ROOT/core_v1/venvs/rootscope_x5_offline_core_v1/bin/python3"

CAMERA='/dev/v4l/by-id/usb-Web_Camera_Web_Camera_202604081837-video-index0'
USB_VID='32e6'
USB_PID='9228'
USB_SERIAL='202604081837'

CAPSULE="$V2_ROOT/core_v1/config/rootscope_x5_offline_core_v1.capsule.json"
MODEL="$CORE/deploy/x5/models/rootscope_seed17_cpu_experimental_opset11.onnx"
REGISTRY="$CORE/app/vision/known_card_template_registry.frozen.experimental.json"
THRESHOLDS="$CORE/app/vision/dual_path_demo.thresholds.example.json"
MATCHER="$CORE/app/vision/card_geometric_matcher.config.example.json"
CALIBRATION="$APP_ROOT/configs/omega/vision_board_replay_new_x5_20260723.json"
PRINT_MANIFEST="$OVERLAY_ROOT/output/pdf/RootScope_A4_four_up_field_cards_20260723_manifest.json"

# 这些摘要属于运行合同，不能从一份临时替换后的文件“现算现填”。
PRINT_MANIFEST_SHA='5e23e6133e9a59d8327bd751c4d6e5434d0c5a86402fd920c9d99e547613d827'
MODEL_SHA='50ae8d2ec1cec0f3748efa127c6b9c00624684f8c112c8a7913c2a0e304a3bad'
REGISTRY_SHA='f5328b66c6b385dd081c1a27577d92b5c63fc18dc00d815f7d3d2561bb68e29f'
CALIBRATION_SHA='e82b196ab627a935fd571bba2b37aa637f040af5ba81fbab5183c1f15aa2e564'
THRESHOLDS_SHA='877205689ad903207e0bcb5ffabdcbc5f1472c00b8f82e72faeb7cdd7d140fcd'
MATCHER_SHA='9952864e50371675e7ea181cc57f2edd9eadd9189bfa0e9eda1e5cdd8f8ca61a'
# runtime capsule 期望值由冻结 v2 模板与下列固定板端绝对路径离线重建，
# 板端只能把现文件与该常量比较，禁止用现文件自算值回填“期望”。
CAPSULE_SHA='1e839948e466895e0b416aababb39402380463f9ce125c8aeb59f4087191cb97'

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_ROOT="$HOME/rootscope_frontend_evidence/$RUN_ID"
install -d -m 700 "$OUT_ROOT"
```

`CAPSULE_SHA` 的冻结重建合同：

- 源模板：`deploy/x5/capsule_config.seed17_cpu_experimental.json`，`2330` bytes，SHA-256=`1b7e9b96ccd4ec4e5ab534e1f305224c3c6330a3ab2efb8ca2e5d0fc52fcfcbb`。
- `project_root` 固定替换为 `/opt/rootscope/.local/share/rootscope-field-v2/core_v1/releases/rootscope_x5_offline_core_v1/rootscope`。
- `python_executable` 固定替换为 `/opt/rootscope/.local/share/rootscope-field-v2/core_v1/venvs/rootscope_x5_offline_core_v1/bin/python3`。
- `model.path` 固定替换为 `/opt/rootscope/.local/share/rootscope-field-v2/core_v1/releases/rootscope_x5_offline_core_v1/rootscope/deploy/x5/models/rootscope_seed17_cpu_experimental_opset11.onnx`。
- 按安装器 `json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"` 编码为 UTF-8；结果必须为 `2765` bytes，SHA-256=`1e839948e466895e0b416aababb39402380463f9ce125c8aeb59f4087191cb97`。
- 板端门禁只执行 `sha256sum "$CAPSULE"` 并与上述常量比较；不允许把板端现文件的自算值写回 `CAPSULE_SHA`。

单帧：

```bash
cd "$APP_ROOT"
PYTHONDONTWRITEBYTECODE=1 "$PY" -m app.omega_vision.uvc_card_frontend \
  --device "$CAMERA" \
  --expected-usb-vid "$USB_VID" \
  --expected-usb-pid "$USB_PID" \
  --expected-usb-serial "$USB_SERIAL" \
  --print-manifest "$PRINT_MANIFEST" \
  --expected-print-manifest-sha256 "$PRINT_MANIFEST_SHA" \
  --mode one-shot --frames 1 --warmup-frames 10 \
  --width 1920 --height 1080 --fps 30 \
  --output-root "$OUT_ROOT" \
  --jsonl "$OUT_ROOT/one_shot.jsonl" \
  --annotated-dir "$OUT_ROOT/one_shot_annotated" \
  --capsule-config "$CAPSULE" \
  --expected-capsule-sha256 "$CAPSULE_SHA" \
  --model-path "$MODEL" \
  --expected-model-sha256 "$MODEL_SHA" \
  --registry "$REGISTRY" \
  --expected-registry-sha256 "$REGISTRY_SHA" \
  --omega-calibration-manifest "$CALIBRATION" \
  --expected-omega-calibration-sha256 "$CALIBRATION_SHA" \
  --thresholds-json "$THRESHOLDS" \
  --expected-thresholds-sha256 "$THRESHOLDS_SHA" \
  --matcher-config-json "$MATCHER" \
  --expected-matcher-config-sha256 "$MATCHER_SHA"
```

有限帧现场观察，例如 12 帧、每 350 ms 一帧：

```bash
cd "$APP_ROOT"
PYTHONDONTWRITEBYTECODE=1 "$PY" -m app.omega_vision.uvc_card_frontend \
  --device "$CAMERA" \
  --expected-usb-vid "$USB_VID" \
  --expected-usb-pid "$USB_PID" \
  --expected-usb-serial "$USB_SERIAL" \
  --print-manifest "$PRINT_MANIFEST" \
  --expected-print-manifest-sha256 "$PRINT_MANIFEST_SHA" \
  --mode bounded --frames 12 --warmup-frames 20 --interval-ms 350 \
  --width 1920 --height 1080 --fps 30 \
  --output-root "$OUT_ROOT" \
  --jsonl "$OUT_ROOT/bounded_12.jsonl" \
  --annotated-dir "$OUT_ROOT/bounded_12_annotated" \
  --capsule-config "$CAPSULE" \
  --expected-capsule-sha256 "$CAPSULE_SHA" \
  --model-path "$MODEL" \
  --expected-model-sha256 "$MODEL_SHA" \
  --registry "$REGISTRY" \
  --expected-registry-sha256 "$REGISTRY_SHA" \
  --omega-calibration-manifest "$CALIBRATION" \
  --expected-omega-calibration-sha256 "$CALIBRATION_SHA" \
  --thresholds-json "$THRESHOLDS" \
  --expected-thresholds-sha256 "$THRESHOLDS_SHA" \
  --matcher-config-json "$MATCHER" \
  --expected-matcher-config-sha256 "$MATCHER_SHA"
```

若相机不能稳定协商 1920×1080，可在**新的一次**运行中同时改为 `--width 1280 --height 720`；不要在同一证据会话中静默切换分辨率。

## 6. 终端显示与退出码

每处理一帧，终端会立即打印一个紧凑 JSON，四个关键字段分别是：

```json
{
  "semantic": "young_tree",
  "omega_ood": "CLASSIFY",
  "geometry_pass_count": 1,
  "final_status": "DISPLAY_ONLY_REGISTERED_CARD_CONSENSUS"
}
```

退出码：

- `0`：请求帧数全部完成，且每帧都得到显示用注册卡共识；
- `2`：请求帧数全部完成，但至少一帧安全拒绝；这是正常的 fail-closed 结果；
- `3`：资产、摄像头身份、采集、推理、标注或释放证据不完整。

无论退出码为何，都只看对应的 `*.final.json` 是否存在并绑定 JSONL。没有 final manifest 的会话属于未完整收口，不能作为现场成功证据。

## 7. 现场摆卡要点

- A4 彩色横向、100% 原尺寸打印后，沿中线裁成四张；正面不写类别文字，背面只写 `G/S/T/U`。
- 卡片保持平整，主体占画面约 35%–70%，避免屏幕翻拍、镜面反光、严重卷曲和过近裁切。
- 三张正类只是已登记打印卡演示；`unknown` 只用于证明拒答，绝不能加入模板注册表。
- 出现 `QUALITY_*`、`OMEGA_OOD_ABSTAIN` 或几何拒绝时，先调整光照、焦距、距离和角度后开启**新会话**，不要修改阈值来追求通过。

## 8. fixture 验证

以下命令只用于 PC 工作区开发验证，不是 X5 immutable overlay 的现场入口；测试不会打开任何真实摄像头：

```bash
cd /path/to/adventurex/rootscope
python -m unittest tests.test_omega_uvc_card_frontend -v
```

它覆盖：四层输出、低质量拒答、unknown 拒答、几何缺失、有限帧与单帧边界、SHA-256 JSONL 链、原子 final manifest、输出根限制、VID/PID/serial 不一致、构造异常释放、四卡 manifest/注册表绑定，以及处理异常后的 `finally` 释放。
