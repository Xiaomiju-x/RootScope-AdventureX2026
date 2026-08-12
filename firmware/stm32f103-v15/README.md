# STM32F103C8T6 RootScope 固件

> 当前待烧录权威版本是 `Z3 + PB6 V15`。完整安全合同、烧录文件和现场标定步骤见
> [`FLASH_V15_交给硬件队友.md`](FLASH_V15_交给硬件队友.md)。
> 本文件下方保留了早期 V12 的机械诊断说明，`MOVE/COIL/MODE` 和无界
> `PUMP,ON` 在 V15 竞赛固件中均被锁定，不得按旧示例操作。

本目录是可加入 STM32CubeMX 生成的 Keil MDK-ARM V5 工程的用户代码。
目标为 STM32F103C8T6，使用 HAL 库、无 RTOS。

STM32 只执行上位机下发的单步命令，不包含植物识别、自动下探、自动定时浇水或
自动回升逻辑。动作顺序和浇水时间全部由地瓜派 RDK X5 控制。

## 1. 已实现功能

- Z 轴（PA0~PA3）控制探针升降的 28BYJ-48 + ULN2003。
- PB6 控制唯一的水泵继电器，PB7 不使用。
- USART1 中断接收上位机命令，建议使用 115200-8-N-1。
- Z 轴执行相对步数命令；X 轴控制已移除。
- 水泵只能由上位机明确发送 `ON` 或 `OFF` 控制，不会自动定时关闭。
- `STOP` 可立即停止 Z 轴电机并关闭唯一水泵。
- 28BYJ-48 使用双相四拍驱动，提高滑轨启动和低速运行扭矩；INA/INB/INC/IND
  按顺序连接时使用标准 `A→B→C→D` 磁相循环，双相模式为
  `AB→BC→CD→DA`。典型输出
  轴一圈约需要 2048 步，实际位移应按滑轨传动机构标定。

> COM13 是 Windows 电脑看到的串口号，不写入 STM32 固件。程序真正运行在
> 地瓜派 RDK X5（Linux）时，端口通常是 `/dev/ttyUSB0` 或
> `/dev/ttyACM0` 一类名称。STM32 只配置 USART1；USB 转串口模块应交叉连接
> TX/RX，并与 STM32 共地，串口逻辑电平应为 3.3 V。

## 2. STM32CubeMX 配置

1. 新建 `STM32F103C8Tx` 工程，Toolchain/IDE 选择 `MDK-ARM V5`。
2. `SYS -> Debug` 选择 `Serial Wire`。
3. 配置下列 GPIO 为 `GPIO_Output`、`No pull`、`Low speed`：

| 功能 | 引脚 | 输出模式 | 上电状态 |
|---|---|---|---|
| Z INA~IND | PA0、PA1、PA2、PA3 | Push-Pull | Low |
| 水泵继电器 1 | PB6 | Open-Drain | 释放（低电平吸合） |

4. 启用 `USART1 -> Asynchronous`：
   - TX = PA9，RX = PA10
   - 115200 bit/s，8 data bits，1 stop bit，No parity
   - NVIC 中启用 `USART1 global interrupt`
5. 时钟按板上实际晶振配置。常见配置是 HSE 8 MHz、SYSCLK 72 MHz，但应以
   实际开发板为准。
6. 生成 Keil MDK-ARM V5 工程。

## 3. 加入 CubeMX 工程

把本目录 `Core/Inc` 中的头文件复制到 CubeMX 工程 `Core/Inc`，把
`Core/Src` 中的源文件复制到工程 `Core/Src`。在 Keil 工程中加入：

- `Core/Src/app.c`
- `Core/Src/stepper.c`

在 CubeMX 生成的 `main.c` 的 USER CODE 区域加入：

```c
/* USER CODE BEGIN Includes */
#include "app.h"
/* USER CODE END Includes */
```

所有 `MX_..._Init()` 执行完后加入：

```c
/* USER CODE BEGIN 2 */
App_Init(&huart1);
/* USER CODE END 2 */
```

主循环加入：

```c
/* USER CODE BEGIN WHILE */
while (1)
{
    App_Task();
/* USER CODE END WHILE */

/* USER CODE BEGIN 3 */
}
/* USER CODE END 3 */
```

在 `main.c` 底部 USER CODE 4 区域加入：

```c
/* USER CODE BEGIN 4 */
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
    App_UartRxCpltCallback(huart);
}

void HAL_UART_ErrorCallback(UART_HandleTypeDef *huart)
{
    App_UartErrorCallback(huart);
}
/* USER CODE END 4 */
```

如果工程中已有同名 HAL 回调，应把 `App_...Callback()` 调用合并到原回调中，
不能重复定义。

## 4. 串口协议

每条命令为大写 ASCII，以 `\r\n` 或 `\n` 结尾，字段用英文逗号分隔。
步数都是相对步数，步进间隔单位为 ms。

| 命令 | 作用 | 示例 |
|---|---|---|
| `PING` | 连通测试 | `PING` |
| `STATUS` | 查询电机及水泵状态 | `STATUS` |
| `STOP` | 停止电机并关闭唯一水泵 | `STOP` |
| `MOVE,Z,步数,间隔` | 探针相对移动，正数向上 | `MOVE,Z,1000,20` |
| `PUMP,编号,ON` | 打开指定水泵 | `PUMP,1,ON` |
| `PUMP,编号,OFF` | 关闭指定水泵 | `PUMP,1,OFF` |
| `VERSION` | 查询当前固件版本 | `VERSION` |
| `IOSTATUS` | 读取 PA0～PA3、PB6 输出锁存状态 | `IOSTATUS` |
| `COIL,Z,模式` | 诊断时直接设置四相位图（0～15） | `COIL,Z,1` |
| `MODE,Z,模式` | 选择电机诊断步序（0～5） | `MODE,Z,0` |

`COIL` 模式按位对应 INA、INB、INC、IND：`1/2/4/8` 分别只吸合 A/B/C/D，
`0` 释放全部线圈。该命令仅用于断开机械负载后的短时间诊断，测试结束必须发送
`COIL,Z,0` 或 `STOP`。

`MODE` 用于适配不同批次电机的实际磁极顺序，设置在复位后恢复默认值 3：

| MODE | 磁极循环 | 驱动方式 |
|---|---|---|
| 0 | A-B-C-D | 单相，电流较小 |
| 1 | A-C-B-D | 单相，电流较小 |
| 2 | A-B-D-C | 单相，电流较小 |
| 3 | A-B-C-D | 双相，扭矩较大（默认） |
| 4 | A-C-B-D | 双相，扭矩较大 |
| 5 | A-B-D-C | 双相，扭矩较大 |

使用例如 `MODE,Z,3` 设置 Z 轴相序。必须等待 Z 轴停止后再切换。
正常带载运行默认使用标准双相模式 3。每次移动先预励磁 150 ms，再从 50 ms 间隔逐步
加速到命令间隔，末步保持 300 ms 后释放。

主要返回：

- `READY,STM32_IRRIGATION`：STM32 上电就绪。
- `ACK,MOVE,Z`：移动命令已接受。
- `DONE,Z`：Z 轴移动完成。
- `ACK,PUMP,1,ON` / `ACK,PUMP,1,OFF`：继电器已经切换。
- `STATUS,Z=0,P=0`：0 表示停止/关闭，1 表示运行/打开。
- `IOSTATUS,PA=0x00,Z=0x0,PB6=1,PLOG=0`：MCU 输出锁存器诊断。
  `PB6` 是输出锁存位（1 表示开漏释放），不是万用表测得的实际电压；
  `PLOG` 是水泵逻辑状态。
- `ERR,...`：参数错误、轴忙或接收溢出。

`PUMP,1,ON` 后水泵会保持打开，直到上位机发送 `PUMP,1,OFF`、发送 `STOP`
或 STM32 复位。STM32 不计算浇水时间，也不会因为旧版二进制心跳缺失而自动改变
上位机已经下达的输出状态。

## 5. 上位机控制示例

识别到植物后的完整流程由上位机逐条控制：

```text
MOVE,Z,3500,20
等待 DONE,Z

PUMP,1,ON
上位机计时 2500 ms
PUMP,1,OFF

MOVE,Z,-3500,20
等待 DONE,Z
```

上位机应等待每条移动命令的 `DONE` 再发送依赖该动作的下一条命令。发生串口断开、
识别程序异常或人工急停时，应立即发送 `STOP`。

## 6. 首次台架测试

1. 先断开水泵动力电源，上电确认唯一继电器不吸合。
2. 当前继电器为低电平吸合；PB6 上电先写 High 并配置为 Open-Drain，
   此时引脚实际为高阻释放，避免 3.3 V 推挽高电平仍让 5 V 继电器输入导通。
3. 保持 Z 轴 INA/INB/INC/IND 分别接 PA0/PA1/PA2/PA3，不要
   再手工交换 B/C；固件默认使用标准 A-B-C-D 双相序列。
4. 发送 `MOVE,Z,64,20`，检查相序及方向。
5. 如果方向相反，把 `Z_POSITIVE_DIRECTION` 改为 `-1`。
6. 接通水泵后使用 `PUMP,1,ON`，观察继电器后立即发送 `PUMP,1,OFF`。
   继电器控制侧 VCC 必须接 5 V，GND 与 STM32 共地，控制输入 IN/D/S 接 PB6；
   如果模块丝印确实为 OUT，必须先查模块说明确认它是输入，不能把两个输出相接。触点侧必须是
   `+5V -> COM`、`NO -> 水泵正极`、`水泵负极 -> 5V负极`，NC 完全不接。

## 7. 当前硬件限制

当前没有原点/限位开关，MCU 不知道滑轨绝对位置，软件步数上限也不能阻止机构撞到
端点。正式运行前建议给 Z 轴增加上下原点/限位开关。

水泵必须使用匹配的独立电源。STM32、ULN2003、继电器控制侧应正确共地；不要用
STM32 的 3.3 V 引脚给电机或水泵供电。

## 8. Keil 编译与烧录

本目录已经包含 CubeMX 生成并接入应用代码的完整 Keil 工程：

```text
MDK-ARM/STM32F103_Irrigation.uvprojx
```

推荐使用 ST-Link 或兼容的 CMSIS-DAP 调试器，通过 SWD 下载：

| 调试器 | STM32F103C8T6 |
|---|---|
| 3.3V / VTref | 3.3V |
| GND | GND |
| SWDIO | PA13 / SWDIO |
| SWCLK | PA14 / SWCLK |
| NRST（推荐） | NRST |

1. 断开水泵动力电源，确认开发板 `BOOT0=0`。
2. 连接 ST-Link 和开发板，调试器参考电平必须是 3.3V。
3. 用 Keil uVision5 打开 `MDK-ARM/STM32F103_Irrigation.uvprojx`。
4. 在 `Options for Target -> Debug` 选择 `ST-Link Debugger`；如果使用
   CMSIS-DAP，则选择对应的 CMSIS-DAP 调试器。
5. 在 `Utilities` 中选择与 Debug 相同的下载器，并勾选
   `Use Target Driver for Flash Programming`。
6. 按 `F7` 编译，确认结果为 `0 Error(s), 0 Warning(s)`。
7. 按 `F8` 或点击 `Download` 下载到 Flash。
8. 下载完成后复位开发板，串口应输出：

```text
READY,STM32_IRRIGATION,2026-07-25-RS-F103-Z-ONLY-PB6-V12,PB6_ACTIVE_LOW_OD,MANUAL,LOAD_BOOST,OUTPUTS_OFF
```

当前工程已使用本机 Keil ARM Compiler 6.21 编译通过，并已生成：

```text
MDK-ARM/STM32F103_Irrigation/FLASH_THIS_Z_ONLY_PB6_V12.axf
MDK-ARM/STM32F103_Irrigation/FLASH_THIS_Z_ONLY_PB6_V12.hex
```

COM13 是运行时串口通信端口，不等同于 SWD 下载器。普通 USB 转串口模块不能按
上述 Keil/ST-Link 步骤烧录。
