# RootScope v3 PC 完成交接根

状态：`PC_COMPLETE_X5_FINAL_CANDIDATE_ACCEPTANCE_PENDING`

RootScope v3 是固定式根区灌溉舱的独立开发根。它不覆盖 v2 回滚基线；
v2 archive SHA-256 仍为
`03ca7b8d9ff8b691f1fd61dc696601ba30f494377a0b2a3cfadb66c19478ed94`。

## PC 侧已经完成

- E0：v2 快照、模型/数据/教师/依赖注册表与五套评测 schema 冻结。
- RootSight：植物证据、OOD/HOLD、before/after 湿润选择性和跨域交叉检查合同。
- RootMind：Qwen2-0.5B Fast + Qwen3-1.7B Deep + 确定性模板，
  `ONE_MODEL_AT_A_TIME`、零物理权限；目标板已经上电并完成外部预验收，
  最终内容寻址候选仍须在 X5 CPU loopback 上重新通过绑定验收。
- Qwen3-1.7B：RTX4050 两阶段 QLoRA、冻结未见 final holdout 32/32、
  Safety Compiler 32 raw accept / 0 fallback / 0 unsafe escape、最终 Q4_K_M。
- RAG2：17 来源、42 chunks、64 gold、36 forbidden；部署选择是
  stdlib-only `bm25_runtime.py`，Dense challenger 未晋级且不打包。
- Action/System：Plant2Action 只生成有上限 proposal；Resource Broker、
  19 类故障矩阵、Decision Receipt、Truth Ribbon 全部 fail-closed。
- 发布：离线 ARM64 wheelhouse、CPU/BPU 模型、Fast/Deep GGUF、资格脚本和
  原子 stage/activate/rollback 工具已齐备。

## PC 资格结果

- Vision：仅冻结静态 CPU 参考 4/4 + 2 个确定性湿润 fixture；不是实时相机或
  本轮 X5 CPU/BPU 对齐。
- RootMind final holdout：JSON、字段、零权限、引用与动作标记五项 32/32；
  16 条真实对抗请求拒绝 16/16。
- Safety Compiler：32/32，fallback=0，unsafe escape=0。
- RAG BM25：PC R@5=`92.19%`，hard top3=`84.09%`，
  forbidden R@5=`94.44%`，citation escape=`0`。
- 物理链：仅 simulation；未打开相机、串口、GPIO 或泵。

## 最终内容寻址候选仍须真实执行

1. 目标板已经上电；最终部署仍必须再次复核固定 SSH alias `rootscope-x5`
   的主机键、hostname、machine-id、板卡序列号、WLAN MAC 与 `aarch64`。
2. 归档上传、StageOnly、板端可信 verifier、离线 wheel bootstrap。
3. CPU ONNX/BM25 回放、canonical `hrt_model_exec` oracle 与 43 样本
   persistent native `libdnn` BPU 回放；Python `hbm_runtime` 路径仅作
   非权威负路径观测，不再作为通过条件。
4. Fast/Deep `llama-server` 临时 loopback smoke；记录加载、TTFT、tokens/s、RSS。
5. 内存/CMA/温度/30 分钟 soak。
6. USB 实时相机、现场偏黄灯光、固定曝光/白平衡、未知卡/OOD。
7. 后续硬件到位后再验 STM32/USB-TTL/ACK/watchdog/急停、称重、湿润和邻格串水。

在这些门禁通过前，禁止声称 v3 已部署、BPU 植物模型已板端资格化、LLM 在
BPU 运行、实时相机已通过或 STM32/水泵物理闭环已完成。

## 上电部署入口

从 `adventurex` PowerShell 执行：

```powershell
.\tools\release_v3\deploy_rootscope_v3_to_x5.ps1 `
  -SshAlias rootscope-x5 `
  -ReleaseDirectory <最终 release 目录> `
  -StageOnly
```

StageOnly 通过后，去掉 `-StageOnly` 执行板端离线软件验收与原子激活。
该验收仍不包含 live camera、资源 soak、STM32 或物理闭环。
