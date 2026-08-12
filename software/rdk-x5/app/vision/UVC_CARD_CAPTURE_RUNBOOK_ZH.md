# RootScope 现场打印卡 UVC 实拍采集说明

## 1. 工具边界

`uvc_card_capture.py` 只负责把一张已经裁开的 A4 四格演示卡，在明确指定的 USB 相机下采集成一组带质量门禁和 SHA-256 的现场光学证据。

- 它不会搜索相机或目录，必须显式给出稳定设备路径、预期 VID:PID、预期序列号、四格打印 manifest 及预期 SHA-256、卡片 ID、卡片角色、输出根目录和全新的直属输出目录。
- 它不会训练模型、加入训练集、修改模板注册表或自动注册类别。
- `unknown` 只能使用 `UNREGISTERED_NEGATIVE_NOT_ACCURACY_HOLDOUT`，不能注册。
- 它不调用串口、GPIO、水泵或 systemd，不拥有灌溉或其他物理执行权限。
- 每份输出都固定标记 `EVENT_OPTICAL_CAPTURE_NOT_AUTO_TRAIN`。即使质量通过，也只表示这次相机采集清晰，不代表准确率、泛化能力或模型合格。

## 2. 已核验相机基线

现场新相机的稳定设备路径为：

```text
/dev/v4l/by-id/usb-Web_Camera_Web_Camera_202604081837-video-index0
```

已核验模式为：

- `MJPG 1920x1080 @ 30 fps`（首选）
- `MJPG 1280x720 @ 30 fps`（USB 带宽或算力降级）

不要改成 `/dev/video0`，因为设备序号在重插或增加其他相机后可能变化。工具只接受 `/dev/v4l/by-id` 的直属 symlink，拒绝 `.`、`..`、嵌套路径和数字设备别名；该 symlink 必须严格解析到 `/dev/videoN` 字符设备。

工具会在打开相机前、打开相机后、释放相机后三次通过只读 sysfs 重新核验下列身份，并要求三次结果完全相同：

- USB VID:PID：`32e6:9228`
- USB serial：`202604081837`
- by-id symlink 的 `lstat`
- `/dev/videoN` 字符设备的 `stat`、major/minor

backend 构造成功后，正常结束、质量拒绝或中途异常都会进入同一释放路径。调用 OpenCV `release()` 本身不算释放成功；工具还必须读回 `isOpened()==false`，再完成第三次稳定 by-id、VID:PID、serial、字符设备及 major/minor 身份核验。静默无效的 `release()`、释放读回失败或释放后身份漂移都会返回运行时错误码 `30`，manifest 中的 `controls.close.release_completed` 只会在释放调用成功且 `isOpened()==false` 时为 `true`，`device_identity.identity_unchanged_across_lifecycle` 只会在三次身份完全一致时为 `true`。若 backend 在构造阶段被 `BaseException` 中断，程序会尝试释放并直接失败退出，但不会生成一份声称完成第三次身份核验的成功 manifest。工具不会使用扫描结果替代上述显式身份。

## 3. 打印与摆放

打印文件：

```text
adventurex/output/pdf/RootScope_A4_four_up_field_cards_20260723.pdf
```

打印时选择 A4 横向、彩色、100% 实际尺寸、关闭“适合页面”，再沿中央竖线和横线裁成四张。每次只在相机前放一张，卡面尽量占画面高度的 60%—80%，避免反光、阴影、手指和其他卡片进入画面。

四张卡固定对应：

| 卡片 ID | 位置 | 显式角色 |
|---|---|---|
| `grass_clump` | 左上 | `REGISTERED_DEMO_REFERENCE_NOT_HOLDOUT` |
| `low_shrub` | 右上 | `REGISTERED_DEMO_REFERENCE_NOT_HOLDOUT` |
| `young_tree` | 左下 | `REGISTERED_DEMO_REFERENCE_NOT_HOLDOUT` |
| `unknown` | 右下 | `UNREGISTERED_NEGATIVE_NOT_ACCURACY_HOLDOUT` |

## 4. 单卡采集命令

先显式创建并确认输出根目录。根目录必须是规范化的现有非 symlink 目录；每次输出目录只能是它的一个直属子目录，必须从未存在。工具使用独占 `mkdir`，拒绝覆盖和嵌套输出。

```bash
OVERLAY_ROOT="$HOME/.local/share/rootscope-event-vision-overlay/releases/rootscope_event_vision_overlay_v1_2"
APP_ROOT="$OVERLAY_ROOT/rootscope"
PRINT_MANIFEST="$OVERLAY_ROOT/output/pdf/RootScope_A4_four_up_field_cards_20260723_manifest.json"
PY="$HOME/.local/share/rootscope-field-v2/core_v1/venvs/rootscope_x5_offline_core_v1/bin/python3"
OUT_ROOT="$HOME/rootscope_event_capture"

cd "$APP_ROOT"
install -d -m 700 "$OUT_ROOT"

PYTHONDONTWRITEBYTECODE=1 "$PY" -m app.vision.uvc_card_capture \
  --device /dev/v4l/by-id/usb-Web_Camera_Web_Camera_202604081837-video-index0 \
  --expected-vid-pid 32e6:9228 \
  --expected-serial 202604081837 \
  --print-manifest "$PRINT_MANIFEST" \
  --expected-print-manifest-sha256 5e23e6133e9a59d8327bd751c4d6e5434d0c5a86402fd920c9d99e547613d827 \
  --card-id grass_clump \
  --class-role REGISTERED_DEMO_REFERENCE_NOT_HOLDOUT \
  --output-root "$OUT_ROOT" \
  --output-dir "$OUT_ROOT/grass_clump_take01" \
  --resolution 1920x1080 \
  --warmup-frames 30 \
  --frames 5 \
  --interval-ms 350
```

依次把 `--card-id`、`--class-role` 和 `--output-dir` 改为另外三张卡。`unknown` 的命令必须写：

```bash
--card-id unknown \
--class-role UNREGISTERED_NEGATIVE_NOT_ACCURACY_HOLDOUT
```

默认保持相机当前曝光和白平衡，不做任何设置。只有画面持续过暗、过亮或明显偏色，而且操作者明确决定时，才可显式使用：

```bash
--exposure-mode manual --exposure-value <相机实际可用值>
--white-balance-mode manual --white-balance-temperature <相机实际可用值>
```

所有显式控制都会记录设置前值、设置确认、读回有效值、比较容差和最终恢复值，并在 `finally` 中恢复；工具不写持久化相机配置。仅有驱动 `set()` 返回成功不算确认，读回值还必须落在明确容差或 V4L2 auto-exposure 等价值内。若设置或恢复无法从读回确认，本次结果 fail-closed。

四格打印 manifest 会被一次性读取为 bytes；同一份 bytes 同时用于 SHA-256 和严格 UTF-8 JSON 解析。除显式 SHA-256 必须匹配外，还要求固定 schema/status、PDF 路径和 SHA、四张卡的固定位置/类别/角色、64 位图片 SHA，以及每张卡 `holdout_claimed=false`、`accuracy_evidence=false`。

## 5. 如何判断结果

每个新目录内包含若干 JPEG 和唯一的 `capture_manifest.json`。

- 进程返回 `0` 且状态为 `EVENT_OPTICAL_CAPTURE_ACCEPTED_NOT_AUTO_TRAIN`：全部帧通过尺寸、亮度、对比度、Laplacian 清晰度和黑白 clipping 门禁。它仍然不能自动进入训练。
- 返回 `20`：帧已保留，但至少一帧质量不合格。调整距离、角度、光线后换一个新目录重拍。
- 返回 `30`：设备身份、采集、I/O、协商、显式控制、恢复或相机释放证明出错。若输出目录已经安全建立，先查看 manifest 的 `error`、`device_identity`、`controls.restore` 和 `controls.close`；只有 `controls.close.release_completed=true`、`controls.close.opened_after_release=false` 且 `device_identity.identity_unchanged_across_lifecycle=true` 才表示完整释放合同通过。
- 返回 `64`：参数、卡片角色、打印 manifest/hash 或输出路径不合法。它与质量拒绝的 `20` 明确区分。

JPEG 会先通过已固定的输出目录 fd、`O_NOFOLLOW|O_EXCL` 写同目录临时文件并 `fsync`，再以 hardlink 原子发布；若同名目标在并发窗口出现，发布失败而绝不覆盖。最终 manifest 使用相同 no-replace 合同最后写入。所有成功路径都不应留下 `.tmp`。如果最终 manifest 因底层文件系统故障无法写入，CLI 仍会输出结构化 I/O 错误和返回码 `30`。

现场确认只看 manifest，不凭肉眼把失败结果改成成功，也不要把 `unknown` 写入任何注册表。
