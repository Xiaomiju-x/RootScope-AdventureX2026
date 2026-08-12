# RootScope r7 BPU shadow runtime

## 冻结结论

本目录是竞赛运行时的**新增、零权限 shadow 链**，不修改任何已有
immutable release，也不改写历史资格结论。

- r7 状态固定为 `SHADOW_CANDIDATE_NOT_DEFAULT`。
- RootScope 旧清单中的 `selected_bin=null` 仍然是权威事实。
- 本代码不会把 r7 `.bin` 复制进旧 field bundle，不会设置
  `selected_bin`，也不会把 BPU logits 接入灌溉决策或动作链。
- worker 和 client 都不提供 TCP、外网、摄像头、串口、GPIO、泵、
  状态机或执行接口。
- fake model 测试只证明协议，不是 BPU 或 X5 真机证据。

当前 r7 本地编译产物：

```text
output/rootscope_bpu_seed17_quant_variant_r7_default_int16_all_nodes/
  model_output/
    rootscope_seed17_resnet18_224x224_rgb_ddr_r7_default_int16_all_nodes.bin
SHA-256:
4dc4bc24741332bb6bc6da184dd1d6f954ae539ba740330490339d76eb200285
```

该 hash 只是把 worker 与明确的 r7 字节绑定，不代表模型已通过旧冻结
数值漂移门、现场精度门、长期稳定性门或默认模型资格。

## 架构

```text
core venv / live vision
  fixed uint8 RGB NCHW [1,3,224,224]
             |
             | AF_UNIX only, 4-byte BE length + strict JSON
             | batch = 1..4, tensor SHA-256, short timeout
             v
BPU system-site CPython 3.10 worker (persistent)
  hash-bind r7 .bin once -> 1..4 sequential batch-1 forwards
             |
             v
  logits [1,4] + per-forward latency + backend truth
  + SHADOW_CANDIDATE_NOT_DEFAULT + exact zero-authority receipt

Any timeout/protocol/hash/shape/authority failure -> CPU fallback
```

输入合同固定为 `uint8 RGB NCHW [1,3,224,224]`。r7 在真机
`hobot_dnn` metadata 中的输出是 `[1,4,1,1]`，部分测试接口是
`[1,4]`；competition worker **只**允许这两个形状，并且只压缩两个
尾部 singleton axis，统一输出 `[1,4]` logits，其他任何维度都拒绝。
每次响应保留真实 input/output metadata。一次请求允许 1 到 4 个
tensor；因为编译图 batch 固定为 1，
worker 在一次请求内顺序执行 1 到 4 次 forward，并分别记录延迟。

真机 pyeasy metadata 的 input properties 为
`dtype=uint8, layout=NCHW, tensor_type=RGB, shape=[1,3,224,224]`，但
其底层 `input.buffer.dtype` 显示为 `int8`。competition adapter 仍然
强制 properties 为 uint8/RGB/NCHW，只把 `int8` 视作已经观察到的
pyeasy runtime storage alias（同时也允许常规 `uint8` buffer），并在
回执中同时记录 declared dtype 与 buffer dtype；AF_UNIX 输入永远保持
`uint8`，不会因底层存储表示而放宽为 int8。

## 未来在 X5 上的显式启动方式

以下只是运行合同示例；本目录创建时未连接 X5、未启动 worker：

```bash
cd /path/to/rootscope-source
/path/to/bpu-system-site-venv/bin/python -m \
  app.competition_runtime.bpu_shadow_worker \
  --socket /run/user/1000/rootscope/r7-bpu-shadow.sock \
  --model-bin /explicit/path/to/r7.bin \
  --expected-model-sha256 \
  4dc4bc24741332bb6bc6da184dd1d6f954ae539ba740330490339d76eb200285
```

模型会在绑定 Unix Socket **之前**完成文件 SHA、单模型、输入和输出
接口检查；任何不匹配都会直接退出。Socket 权限为 `0600`。

core venv 只需构造 `BpuShadowClient` 并传入现有 CPU ONNX 回调：

```python
from app.competition_runtime import BpuShadowClient

client = BpuShadowClient(
    "/run/user/1000/rootscope/r7-bpu-shadow.sock",
    timeout_s=0.18,
)
receipt = client.infer_tensors(tensors, cpu_fallback=run_existing_cpu_onnx)
```

`receipt["status"]` 只能是：

- `BPU_SHADOW_OK`：收到通过 hash、协议、结果溯源和零权限检查的
  shadow logits；
- `CPU_FALLBACK_OK`：BPU shadow 任一环节失败，现有 CPU 路径成功；
- `CPU_FALLBACK_FAILED_NO_RESULT`：两条只读推理路径都没有结果，仍然
  不产生任何动作。

## 定向测试

```bash
python -m pytest -q tests/test_competition_bpu_shadow.py
```

测试使用 fake model 覆盖 1/4 tensor 协议、截断帧、错误 shape、短超时、
模型 hash、响应结果 hash 和 exact zero-authority；fake 后端在所有响应
中均明确标为 `FAKE_MODEL_PROTOCOL_TEST_NOT_BPU_EVIDENCE`。
