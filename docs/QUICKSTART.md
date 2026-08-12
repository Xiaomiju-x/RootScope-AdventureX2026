# 快速开始 / Quickstart

本指南先复现**完全不连接设备**的安全决策参考层，再说明如何只读检查完整工程。不要从这里直接跳到通电、烧录或运动。

## 1. 环境

- Python 3.10 或更高版本；CI 覆盖 Python 3.10 和 3.12。
- Git；如需取得模型大文件，还需要 Git LFS。
- Linux、macOS、Windows 均可运行根目录参考层。
- RDK X5 板端运行和 STM32 构建分别见 [RDK X5 部署](RDK_X5_DEPLOYMENT.md) 与 [STM32 构建](STM32_BUILD.md)。

## 2. 克隆与安装

```bash
git clone https://github.com/Xiaomiju-x/RootScope-AdventureX2026.git
cd RootScope-AdventureX2026
git lfs pull                 # 若当前版本使用 LFS；没有 LFS 对象时可跳过
python -m venv .venv
```

激活虚拟环境：

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

安装根目录参考包和测试依赖：

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## 3. 运行三个安全夹具

```bash
rootscope-public examples/grass_agree.json
rootscope-public examples/conflict_hold.json
rootscope-public examples/ood_hold.json
```

你应看到：

- 一致且新鲜的合成证据可以产生抽象 `action_tier`；
- 几何/语义冲突或 OOD 输入返回 `HOLD`；
- 所有结果均为 `proposal_only: true`；
- `hardware_command` 始终是 `null`；
- 每份输出包含基于规范化 JSON 计算的 SHA-256。

这些夹具不代表现场精度，也不会加载竞赛模型。

## 4. 测试与公开发布审计

```bash
pytest
python tools/audit_public_release.py
```

测试验证 HOLD 优先级、LLM 零动作权限和确定性回执。发布审计用于发现凭据模式、私有网络身份、用户绝对路径以及不应直接进入 Git 的构建产物。审计通过不等于代码或硬件已经过安全认证。

## 5. 只读浏览完整工程

```bash
python -m compileall src research/rootscope_v3 software/rdk-x5/app
python -m pytest research/rootscope_v3/tests
```

`software/rdk-x5/tests` 包含 Linux/X5 专用模块及较重依赖，建议在 Ubuntu 或 RDK OS 中单独创建环境：

```bash
python -m venv .venv-x5
source .venv-x5/bin/activate
python -m pip install -e "software/rdk-x5[hardware]"
python -m pip install pytest
pytest software/rdk-x5/tests
```

不要在普通 PC 上尝试导入 Horizon/Bayes-e 板端运行时。BPU 复现需使用与模型卡声明一致的 RDK OS 和工具链。

## 6. 下一步

| 目标 | 文档 |
|---|---|
| 理解证据与权限分离 | [ARCHITECTURE.md](ARCHITECTURE.md) |
| 搭建机械/电气系统 | [HARDWARE_WIRING.md](HARDWARE_WIRING.md)、[MECHANICAL.md](MECHANICAL.md) |
| 配置板端软件 | [RDK_X5_DEPLOYMENT.md](RDK_X5_DEPLOYMENT.md) |
| 构建 V15 固件 | [STM32_BUILD.md](STM32_BUILD.md) |
| 获取/校验模型 | [MODEL_ASSETS.md](MODEL_ASSETS.md) |
| 按证据等级复现 | [REPRODUCIBILITY.md](REPRODUCIBILITY.md) |

## English summary

The root package is a device-free reference implementation. Install with `python -m pip install -e ".[dev]"`, run the three JSON fixtures, then run `pytest` and `python tools/audit_public_release.py`. It cannot open a camera, serial port, GPIO, pump, or network-control endpoint. Full board and firmware reproduction is intentionally separated into gated guides.
