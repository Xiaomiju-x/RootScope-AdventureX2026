# RDK X5 部署 / RDK X5 Deployment

本指南把公开源码部署到一台**由复现者自己管理**的 RDK X5。它不包含比赛设备地址、设备唯一身份、账号或自动连接脚本，也不会修改网络。先完成 PC 级复现，再按只读 → 无水 → 单动作顺序晋级。

> 本项目已冻结为获奖档案。以下内容用于复现，不是远程操作原比赛设备的入口。

## 1. 前置条件

- RDK X5 与官方支持的系统镜像；
- Python 3.10 运行环境；
- 与目标模型匹配的 D-Robotics/Bayes-e 运行时（仅 BPU 路径需要）；
- UVC 相机；
- 可选：3.3 V USB–TTL 与一块**由你自己构建**的 STM32F103 V15；
- 本仓库、Git LFS（如当前 release 使用 LFS）和足够的只读存储空间。

不要在板上安装训练栈。训练、数据抓取和模型转换放在 PC；X5 只做推理、证据、展示和受控执行。

## 2. 获取与校验

```bash
git clone https://github.com/Xiaomiju-x/RootScope-AdventureX2026.git
cd RootScope-AdventureX2026
git lfs pull
git status --short
python tools/audit_public_release.py
```

随后依据 [`model-assets/MANIFEST.json`](../model-assets/MANIFEST.json)（若该 release 提供）逐个校验 SHA-256。不要运行哈希不匹配、来源不明或针对不同 BPU 工具链构建的文件。

## 3. CPU 运行环境

```bash
python3 -m venv .venv-x5
source .venv-x5/bin/activate
python -m pip install --upgrade pip
python -m pip install -e "software/rdk-x5[hardware]"
python -m pip install pytest
```

板端 OpenCV、ONNX Runtime 和相机依赖可能由系统镜像提供。优先使用与板卡系统兼容的 wheel/系统包，不要盲目升级系统级运行时。

先运行设备无关测试：

```bash
pytest software/rdk-x5/tests/test_probe_depth.py
pytest software/rdk-x5/tests/test_competition_llm.py
pytest software/rdk-x5/tests/test_rootmind_v3.py
```

## 4. 相机只读资格

1. 用 `v4l2-ctl --list-devices` 找到复现者自己的相机。
2. 以 `/dev/v4l/by-id/` 稳定别名记录本机配置；不要把真实别名和 USB 拓扑提交到 Git。
3. 列出支持的格式、分辨率和帧率；选择相机真实支持的组合。
4. 固定支架、光线、曝光和白平衡后，再生成模板或验收输入。
5. 先使用只读视觉入口，不连接 STM32 和水泵。

固定证据包可通过视觉-only 工具回放：

```bash
python software/rdk-x5/tools/x5_answer_card_validate.py \
  --bundle <ABSOLUTE_PATH_TO_YOUR_VERIFIED_BUNDLE>
```

工具只读取图像和模型，但 `<...>` 必须替换为复现者本机路径，不能原样执行。结果只适用于该固定证据包。

## 5. BPU 固定输入资格

1. 核对模型卡中的芯片、工具链、输入布局、颜色顺序、量化与输出合同。
2. 先在 PC 验证转换输入和 SHA，再把内容寻址目录复制到 X5。
3. 使用板端 `hrt_model_exec` 对冻结输入运行 canonical 回放。
4. 再运行持久加载路径，检查加载一次、多样本、干净退出和恢复。
5. 把 FP32/CPU 与 BPU 输出比较写入机器可读回执。

公开的 43/43 和 mean cosine 1.0 是特定冻结样本的既有回放结果。新的硬件、系统镜像或模型文件必须生成新的回执，不能沿用旧数字。

## 6. RootMind / RAG

- Fast、Deep 和 BM25/HOLD 是按需角色，不要同时常驻。
- LLM 服务只绑定 loopback，不开放局域网或公网控制接口。
- 只接受严格 JSON/结构化输出；解析失败、超时或无引用时退回确定性模板。
- LLM 进程不能拥有串口、GPIO 或执行器组权限。
- 模型切换前后记录内存与文件页缓存；仅释放本进程/模型对应缓存，不执行全局 `drop_caches`。

## 7. STM32 只读资格

执行器动力保持断开：

1. 为复现者自己的 USB–TTL 创建稳定 udev 别名；规则中使用本机核验的 `<VID>`、`<PID>`、`<INTERFACE>`/`<PORT_PATH>` 占位值。
2. 复核设备路径的所有者和串口组权限；不要使用 `sudo` 运行完整应用。
3. 仅查询 `VERSION`、`STATUS`、`IOSTATUS` 和固件身份。
4. 期望 V15、build `2026072515`、输出关闭、控制器锁存、无活动任务。
5. 任何字段不一致即停止；不要修改上位机期望值迁就未知固件。

公开仓中的预检工具必须先复制其 example 配置，在本机填写设备身份；真实值保持未跟踪。不要直接使用历史比赛配置。

## 8. 晋级阶梯

| 门 | 相机 | STM32 | 执行器动力 | 目标 |
|---|---|---|---|---|
| G0 软件测试 | 否 | 否 | 断 | 合同/测试通过 |
| G1 固定输入回放 | 否 | 否 | 断 | CPU/BPU 输出可解释 |
| G2 实时视觉 | 是 | 否 | 断 | 四卡、未知、遮挡/OOD |
| G3 MCU 只读 | 可选 | 只读 | 断 | 身份和安全态 |
| G4 协议 dry-run | 否 | 是 | 断 | 心跳丢失与 STOP 回到关闭 |
| G5 单执行器 | 否 | 是 | 单路限流 | 方向、超时、最终关闭 |
| G6 完整受控演示 | 是 | 是 | 接通 | 一次事务、人工监护、证据封存 |

每一级都保存新的时间戳、软件版本、模型哈希和最终安全状态；失败时不能自动进入下一级。

## 9. 启动策略

调试与复现阶段只在前台运行。不要在未完成 G0–G6 前创建 `systemd enable`、开机自动解锁或无人值守服务。即使后续创建服务，也应满足：

- 默认没有物理动作权限；
- 配置和模型是只读、内容寻址版本；
- 启动先核验身份与输出关闭；
- 退出、崩溃和超时都触发下位机安全关闭；
- 相机/LLM/UI 失败不能阻断 STOP；
- 远程网页只有只读证据，不反向控制设备。

## 10. 卸载与恢复

停止前台进程后，独立读取 STM32 最终状态并物理确认泵停止、线圈释放。移除候选版本应只删除复现者创建的独立目录，不能覆盖系统运行时或其他项目。网络、Wi-Fi、路由、VPN 与防火墙不属于本部署脚本的修改范围。

## English summary

Deploy in gates: device-free tests, frozen-input CPU/BPU replay, live camera without actuators, read-only STM32 identity, dry-run with actuator power disconnected, then supervised single-action and wet tests. Use your own local device aliases and configuration; never commit board identity or credentials. Keep LLM/RAG loopback-only and without serial/GPIO permissions. Do not enable startup services before all gates pass.
