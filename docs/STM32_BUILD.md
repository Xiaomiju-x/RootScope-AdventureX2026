# STM32 V15 构建与烧录 / STM32 V15 Build & Flash

最终固件位于 [`firmware/stm32f103-v15/`](../firmware/stm32f103-v15/)，目标为 STM32F103C8T6，使用 STM32 HAL、无 RTOS。主仓保留 `.ioc`、用户源码、HAL/CMSIS 依赖和测试；生成的 IDE 缓存、对象文件与本机设置不作为源码提交。

> 烧录前断开水泵和步进电机动力。编译成功不授权运动或注水。

## 1. V15 冻结合同

| 项 | 值 |
|---|---|
| version | `2026-07-25-RS-F103-Z3-PB6-V15` |
| build ID | `2026072515` |
| build tag | `rs-f103-z3-pb6-v15` |
| hardware variant | `2` |
| capability mask | `0x00000079` |
| heartbeat timeout | `1000 ms` |
| pump duration limits | `100–30000 ms` |
| Z presets | `0 / 1024 / 1536 / 2048` steps |

关键安全属性：

- `MOVE`、`COIL`、`MODE` 和无界 `PUMP,ON` 被锁定；
- 非零 `DEPTH` 需要新鲜心跳、解锁和一次性人工 `ZHOME` 确认；
- 一次确认只允许一次下降；
- 水泵只能进入有时限的任务；
- STOP、心跳丢失、UART 故障、硬超时或复位会关闭 PB6、释放 PA0–PA3 并锁存；
- 没有自动回升。

权威常量在 [`Core/Inc/app_config.h`](../firmware/stm32f103-v15/Core/Inc/app_config.h)，不要只根据文档复制数值。

## 2. 工具链

竞赛归档使用 STM32CubeMX/STM32CubeF1 和 Keil MDK-ARM，最终记录的 Arm Compiler 为 6.13.1，构建结果为 `0 Error(s), 0 Warning(s)`。不同 Cube/HAL/编译器版本可能产生不同 HEX；复现者应记录精确版本，而不是假定哈希相同。

可选工具：

- STM32CubeMX：打开 `STM32F103_Irrigation.ioc`；
- Keil µVision / Arm Compiler 6：生成并构建 MDK-ARM；
- ST-Link 或 CMSIS-DAP：SWD 下载；
- `arm-none-eabi-*`：仅当复现者自行补齐等价构建系统并验证链接脚本时使用。

## 3. 生成工程

1. 复制仓库到新的构建目录，保持源码树干净。
2. 用 CubeMX 打开 `.ioc`，确认目标 `STM32F103C8Tx`、USART1、PA0–PA3、PB6 与 SWD。
3. 选择 MDK-ARM，生成工程。
4. 生成前后对 `Core/` 与 `Drivers/` 做 Git diff；CubeMX 版本变化不得静默覆盖用户逻辑。
5. 确认 `app.c`、`stepper.c` 已加入编译，HAL UART 回调仍调用应用回调。

不要把开发者本机路径、调试器序列号、IDE 用户配置或构建输出提交到 Git。

## 4. 编译与检查

在 µVision 打开生成的工程并执行 Rebuild。至少检查：

- 目标芯片和 Flash/RAM 布局正确；
- 没有 warning/error；
- map 文件中没有意外的旧版应用符号；
- `APP_FIRMWARE_BUILD_ID` 和版本字符串为 V15；
- GPIO 初始化先写安全电平，再切换模式；
- PB6 是开漏且默认释放；
- PA0–PA3 空闲时全低。

对输出计算哈希：

```bash
sha256sum <PATH_TO_V15_HEX>
```

竞赛冻结 HEX 的归档 SHA-256 为：

```text
5016b96d138d4ffad2088dd5da288b4d68c5deba781555ad82eb6f7fb4bfd887
```

只有在工具链、源码提交、链接配置和输出语义完全一致时才期待相同哈希。否则记录新的哈希和差异，不要重命名为官方 V15。

## 5. 主机侧测试

`firmware/stm32f103-v15/Tests/` 保存协议/逻辑测试材料。还应在不接执行器的硬件上覆盖：

- 上电默认关闭；
- 未解锁时拒绝档位/泵任务；
- 无人工回顶确认时拒绝下降；
- 一次确认后拒绝第二次下降；
- 心跳中断后在合同时间内锁存；
- 无界开泵命令被拒绝；
- 定时任务在完成、超时和 UART 故障后均关泵；
- STOP 具有最高优先级。

## 6. SWD 烧录

| 调试器 | STM32F103 |
|---|---|
| VTref | 3.3 V |
| GND | GND |
| SWDIO | PA13 |
| SWCLK | PA14 |
| NRST（推荐） | NRST |

烧录顺序：

1. 断开泵和步进电机动力，`BOOT0=0`；
2. 连接 SWD，只给逻辑域上电；
3. 擦除/下载/校验 Flash；
4. 复位并观察启动行；
5. 仅执行版本和状态查询；
6. 确认 `P=0`、锁存、无任务、PA0–PA3 释放、PB6 关泵；
7. 保存构建哈希与只读回执；
8. 在独立的硬件验收中才考虑执行器动力。

普通 USB–TTL 只用于运行时串口，不能代替 SWD 烧录器。

## 7. 不要做的事

- 不要因为方向错误而在线交换相线或放宽自由运动命令；断电后核验机械和线序。
- 不要将 1024/1536/2048 称为毫米、厘米或根深。
- 不要为调试加入“自动回升”；最终机构没有限位与绝对位置。
- 不要移除心跳、硬超时、锁存或一次性 ZHOME 门。
- 不要在泵接通时做首次烧录或固件身份核验。

## English summary

Build the STM32F103 V15 project from the committed CubeMX/HAL sources in a clean directory. The archived competition build used Arm Compiler 6.13.1 and produced the recorded HEX hash above, but different toolchains may differ. Flash over SWD with motor/pump power disconnected, then perform identity and outputs-off checks only. V15 forbids arbitrary motion and unbounded pump-on commands, requires heartbeat and one-shot manual-home confirmation, and fails closed on STOP, timeout, UART fault, or reset.
