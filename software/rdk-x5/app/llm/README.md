# RootScope 本地 LLM 只读讲解层

这条链只负责把已经结构化的视觉、湿润与安全证据讲清楚。它不是植物识别器、控制器或安全门，也不能改变 RootScope 状态机。

`read_only_explainer.py` 的硬边界：

- 只允许回环 HTTP 端点；启用后先把 `localhost` 解析结果全部核成 loopback，再直接连接数值 loopback，HTTP 重定向永不跟随；
- 不提供 tool call、串口、水泵、状态机或文件写入接口；
- prompt 与输入快照分别落 SHA-256；
- 只接受严格 JSON，authority 六项必须全部为 `false`；
- 启用时必须给出冻结 GGUF SHA-256；输出含控制指令、字段越界、解析失败、超时或服务离线时，退回确定性证据摘要；降级摘要不会原样回显不可信命令文本；
- LLM 输出永远是 `EXPLANATION_ONLY`，不能提升感知 `qualified`，也不能证明 BPU、X5 或物理闭环完成。

已把 XRD 的 `qwen2_05b_distill.Q4_K_M.gguf` 机械复制到 AdventureX 的只读 staging release。复制件为 `397805120` 字节，SHA-256 为 `6c8adc95de81b1d0103fcf900a5f369d5936d234e42ed64881dad56e0104e77b`。这只证明字节级复用完成，不证明 llama.cpp 已随包提供、X5 可运行或时延合格。项目知识由结构化证据和固定 prompt 提供，不临时微调一个未经验证的“沙漠专家”。

离线禁用模式自测（不会打开网络套接字）：

```bash
cd /opt/rootscope/current
/opt/rootscope/venv/bin/python3 -m app.llm.read_only_explainer \
  --snapshot-json evidence/example_snapshot.json
```

禁用 CLI 只写 stdout。启用必须走 `deploy/x5/scripts/explain_readonly_snapshot.py`，它先复核 release、GGUF、外部 llama-server 哈希和 `/health`，随后才请求回环端点。返回码 `0` 仅代表模型输出通过只读 JSON 合同；返回码 `2` 代表已安全降级为确定性摘要，二者都没有执行权限。
