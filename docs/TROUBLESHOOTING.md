# FAQ 与故障排查 / FAQ & Troubleshooting

先判断问题属于软件、模型、相机、串口、固件、机械还是水路。任何物理异常都先切断执行器动力并确认泵关闭、线圈释放；不要边运动边修改参数。

## 项目与结果

### RootScope 是移动机器人吗？

不是。最终作品是固定式根区灌溉舱。照片中的 Scout Mini 轮式底盘只作承载架和供电底座；决赛控制链没有轮式运动、SLAM、Nav2 或导航。

### 为什么叫“根区”却不测根深？

名称描述目标应用方向。比赛原型使用固定答辩卡的可见形态，映射到三个演示步数档位；没有地下根系传感或生物根深模型。步数不能换算为根深。

### 银奖第二名与“金奖”冲突吗？

项目正式表述是 AdventureX 2026 D-Robotics「Give AI a Body」赛道**银奖、最终第 2 名**。仓库不自行推导其他奖项等级。

### 四卡 4/4 是不是 100% 精度？

不是开放世界精度。它只说明四张固定打印卡在受控现场资格中通过。自然场景、不同相机/灯光/打印机需要新的独立评测。

## 安装与测试

### `rootscope-public` 命令不存在

确认虚拟环境已激活，并在仓库根目录运行：

```bash
python -m pip install -e ".[dev]"
python -m rootscope_public.cli examples/grass_agree.json
```

### `pytest` 找不到包

不要直接从未安装的源码目录运行。执行 editable install，或确认当前解释器与安装时相同：

```bash
python -c "import sys; print(sys.executable)"
python -m pip show rootscope-public
python -m pytest
```

### Git LFS 文件只有几行指针文本

```bash
git lfs install
git lfs pull
git lfs ls-files
```

然后校验 manifest 中的文件大小和 SHA-256。无权访问 LFS 对象时仍可运行根目录 device-free 参考层，但不能声称完成模型复现。

### 发布审计报私有 IP/路径

不要把该模式加入允许列表来“通过”。定位文件，将真实地址改为 `<DEVICE_ADDRESS>`、RFC 5737 文档地址或环境变量；receipt 中删除用户名、绝对路径、MAC、machine-id、boot-id 和序列号。测试恶意模式时应使用显式 fixture 标记。

## 相机与视觉

### 相机打不开

1. `v4l2-ctl --list-devices` 确认设备存在；
2. 检查当前用户是否有权限及相机是否被其他进程占用；
3. 使用 `/dev/v4l/by-id/` 本机别名，不假定 `video0`；
4. 检查相机支持的格式/分辨率/帧率；
5. 先用系统相机工具验证，再运行 RootScope。

不要扫描网络相机或修改网络来替代缺失的本地 UVC 设备。

### 四张卡突然都失败

检查相机/支架/灯光/纸张/打印比例/曝光/白平衡是否变化、镜头是否脏污、ROI 是否被探针或反光遮挡。光学条件变化意味着原模板资格失效，应重新采集和评测，不能先降低门限。

### CNN 正确但结果 HOLD

这是预期的 fail-closed 行为。查看几何复核、质量、OOD、新鲜度和设备安全态；高 CNN 置信度不能覆盖独立证据失败。

### BPU top-1 一致但数值不同

比较相同预处理、量化尺度、输出顺序和完整张量，不只比较 top-1。记录 runtime/工具链/模型哈希。如果固定合同容差不通过，保留失败回执，不能把模型标为已验收。

## LLM / RAG

### LLM 输出不是合法 JSON

退回 BM25/HOLD 确定性模板并记录解析失败。不要用正则从自由文本“抢救”动作参数，也不要重试到得到想要的答案。

### 检索没有引用

检查 corpus、allowlist 与 index manifest 哈希是否一致。无允许引用时拒答/降级；禁止让 LLM 用记忆补充并标成有据答案。

### 板上内存不足

确认只加载一个 RootMind 角色，关闭当前模型后再切换；使用模型卡规定的量化版本。不要关闭桌面/系统服务或执行全局 `drop_caches` 来掩盖资源问题。

## 串口与 STM32

### 串口设备号变化

为你自己的适配器建立 udev 稳定别名，条件来自本机 `udevadm info`。不要复制比赛设备 ID，也不要依赖 `ttyUSB0`。TX/RX 交叉、3.3 V TTL、共地且不接 VCC 反向供电。

### 版本或 capability 不一致

停止。核验构建与烧录记录；不要修改上位机期望值迁就未知固件。执行器动力保持断开，重新烧录/只读资格按 [STM32_BUILD.md](STM32_BUILD.md) 执行。

### 控制器一直锁存

锁存是安全状态。先排除 STOP、心跳、UART、硬超时、上次任务、输出和机械问题，再按合同显式恢复。不要添加自动解锁或启动即解锁。

### 心跳丢失

确认主机进程调度、串口线和供电稳定。固件应在超时后关泵、释放线圈并锁存；先验证该状态。通信恢复后仍需新的人工检查和新事务，不能自动续跑。

## 机械与水路

### 探针方向相反或卡滞

立即 STOP 并切断电机动力。检查 INA–IND 顺序、齿条啮合、导轨平行、软管余量和手动行程。不要在线交换线缆、自动反向脱困或加大步数。

### 1536/2048 步超时

把它当作失败并保持锁存。记录实际位移、声音、温升和卡滞位置；人工回顶后才可诊断。增加超时可能扩大机械损坏，不能作为第一修复。

### MCU 显示关泵但仍滴水

MCU 状态只说明控制输出。检查继电器触点、软管残水、虹吸、出水头高度和阀/泵机械状态。物理关闭必须由独立观察确认。

### 继电器上电误吸合

立刻断开泵动力。核对 PB6 开漏、低有效、初始化顺序、模块控制电平和控制侧供电。首次检查使用假负载；没有证明默认释放前不能接泵。

## 如何报告问题

普通问题使用 issue 模板并附：commit、平台/工具版本、最小复现、期望/实际、已清理日志和证据等级。不要上传凭据、真实网络拓扑、设备唯一身份或人物原始素材。安全漏洞使用 [SECURITY.md](../SECURITY.md) 的私下渠道。

## English summary

Fail closed first: disconnect actuator power and confirm pump/motor outputs are off before troubleshooting. Do not lower vision gates, accept unknown firmware, automatically unlock/retry, extend mechanical timeouts, or treat MCU state as observed water flow. Reports should include the commit, exact tool versions, minimal reproduction, sanitized logs, and evidence level—never credentials or device identity.
