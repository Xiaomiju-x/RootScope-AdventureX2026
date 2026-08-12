# XRD → RootScope clean-X5 只读复用盘点

盘点日期：2026-07-17  
边界：仅查看 `xrd_backup` 本地文件；未 SSH、未探测设备、未改网络、未启动服务。

| XRD 模式 | RootScope 采用方式 | 明确不采用 |
|---|---|---|
| `x5_runtime_preflight.sh` 的 PASS/WARN/FAIL、显式路径和 SHA-256 | 重写为 RootScope 独立 schema；只检查显式文件、模块元数据和设备别名是否存在 | ROS 包、XRD 模型哈希、自动服务状态推断、设备扫描 |
| systemd 的 `EnvironmentFile`、失败重启和资源上限 | 形成锁定 Dashboard 与可选只读 LLM 两个独立模板 | XRD 服务名、材料预测端口、ROS/TROS 启动链、开机即获得硬件权限 |
| ONNX Runtime 显式 `CPUExecutionProvider` | 强制 CPU-only session、模型哈希、静态 `1x3xHxW` 合同与确定性模拟输入 | 任意 provider fallback、把 CPU 一致性写成 BPU/X5/精度证据 |
| UVC 明确设备路径、低缓存、延迟打开 | capsule 仅声明 `/dev/rootscope_uvc` 和 `/dev/rootscope_depth` 可选别名，预检只做单路径存在性检查 | XRD 中 `0,8,1,4` 自动试探、当前服务打开摄像头、把深度帧隐式当 RGB 分类输入 |
| llama.cpp 独立服务、GGUF 和内存护栏 | 将最小的 XRD Qwen2 0.5B GGUF 机械复制进 AdventureX staging；提供无硬编码账号、无自动 enable、双人工 gate、回环防火墙和严格哈希预检的用户级只读模板 | XRD 材料语料、1 GB 级替代模型、车载 Agent/ROS 工具、`0.0.0.0` 监听、LLM 控泵 |

结论：主要复用工程方法；唯一机械复用的模型字节是只读讲解用 0.5B GGUF，不继承 XRD 材料领域能力或真机结论。RootScope 当前 ONNX 仍是机器筛选数据上的实验产物；seed17 CPU 配置虽已哈希绑定，`model_candidate/model_qualified/x5_validated/bpu_ready` 仍全部为 false。
