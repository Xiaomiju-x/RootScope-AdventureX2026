# RootScope Z3 + PB6 / V15 烧录交接

## 结论

本次烧录 V15。它保留 V14 的单向下降、人工回顶、单次下降锁、心跳和断联释放，
只冻结现场确认后的最终四状态映射：

| 识别状态 | 固件档位 | 步数 | 动作 |
|---|---:|---:|---|
| `non_target` / 纯沙 | 0 | 0 | HOLD，不运动 |
| `grass_clump` / 草丛 | 1 | 1024 | 向下，最浅植物档 |
| `low_shrub` / 灌木 | 2 | 1536 | 向下，中档 |
| `young_tree` / 幼树 | 3 | 2048 | 向下，深档 |

这些是步数档位，不是毫米、厘米或真实生物根深。现场已经观察到 V14 的 1024 步
向下运动且深度合适，因此把它冻结为 V15 的最浅植物档；1536/2048 步仍需逐档测试。

## 唯一应烧录文件

```text
MDK-ARM/STM32F103_Irrigation/FLASH_THIS_Z3_PB6_V15.hex
```

绝对路径：

```text
firmware/stm32f103-v15/release/FLASH_THIS_Z3_PB6_V15.hex
```

SHA-256：

```text
5016B96D138D4FFAD2088DD5DA288B4D68C5DEBA781555AD82EB6F7FB4BFD887
```

调试符号文件：

```text
MDK-ARM/STM32F103_Irrigation/FLASH_THIS_Z3_PB6_V15.axf
SHA-256=F7D7A714B017064864EF63AD1A42A28A1F6411355501351577CBC546DCEDF29B
```

Keil Arm Compiler 6.13.1 构建结果：

```text
Code=23856  RO-data=1756  RW-data=16  ZI-data=2288
0 Error(s), 0 Warning(s)
```

## V15 身份

```text
version    = 2026-07-25-RS-F103-Z3-PB6-V15
build_id   = 2026072515
build_tag  = rs-f103-z3-pb6-v15
variant    = 2
caps       = 0x00000079
```

预期启动行：

```text
READY,STM32_IRRIGATION,2026-07-25-RS-F103-Z3-PB6-V15,PB6_ACTIVE_LOW_OD,Z3_DOWN_ONLY,MANUAL_RETURN,BOOT_LOCKED,OUTPUTS_OFF
```

## 烧录后立即检查

1. 烧录完成后不要直接测试电机或水泵。
2. USB 转 TTL 接回 RDK X5。
3. 回复队长：`V15 已烧录`。
4. 先由 X5 只读核验：
   - V15 / build `2026072515`
   - `P=0, LOCK=1`
   - `ZHOME=0, ZUSED=0`
   - `PA0～PA3=0`
   - `PB6=1, PLOG=0`
5. 只读核验通过后，才可在人工回顶和现场安全确认下逐档测试。

## 不变的安全边界

- `MOVE / COIL / MODE` 仍锁死。
- 非零 `DEPTH` 需要新鲜心跳、解锁和 `ZHOME,CONFIRM`。
- 一次人工回顶确认只允许一次下降。
- 不自动回升；由现场人员手动回顶。
- `STOP`、心跳丢失、UART 故障或复位会释放 PA0～PA3，并关闭 PB6。
- 水泵只能走有时限的二进制任务，不能使用无上限 `PUMP,ON`。
