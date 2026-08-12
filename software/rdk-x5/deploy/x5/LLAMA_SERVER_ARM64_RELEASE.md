# RootScope llama-server aarch64 发布候选

已在 AdventureX 工作目录内从官方 `ggml-org/llama.cpp` 的 `b9637` 标签、提交 `aedb2a5e9ca3d4064148bbb919e0ddc0c1b70ab3` 构建 Ubuntu 22.04/aarch64 候选：

- 目录：`output/rootscope_llama_server_arm64_b9637_v1`
- 离线归档：`output/rootscope_llama_server_arm64_b9637_v1.tar`
- `llama-server` SHA-256：`dcb636215243b8911488b8ca96f0c39bedee14e92f44f7d0ef6c599419acf9b9`
- 归档 SHA-256：`48f2048a9e207ff4215c8867447a8546ac9f438705731b20f6d2905440a167c2`

构建使用 Ubuntu 22.04 AMD64 容器中的 GCC 11 aarch64 交叉工具链，CPU 指令基线固定为保守的 `armv8-a`。OpenMP、OpenSSL/HTTPS、动态后端和共享库均关闭，`libstdc++`/`libgcc` 静态链接；ELF 只需要 Ubuntu 22.04 自带的 `libm.so.6`、`libc.so.6` 和 `ld-linux-aarch64.so.1`，最高 GLIBC 符号版本为 `2.34`，无 RPATH/RUNPATH。

Ubuntu 22.04 arm64 QEMU 已在 Docker `--network none` 下完成：

- `llama-server --version`；
- `ldd` 无缺失依赖；
- 加载已锁定的 Qwen2 0.5B GGUF；
- loopback `/health` 返回 200；
- loopback `/v1/chat/completions` 返回 200 和非空最小结果。

这些证据严格属于 `CROSS_BUILD_QEMU_SMOKE_NOT_X5_VALIDATION`。它不证明 RDK X5 真板时延、温度、长期稳定性或服务资格。现场必须先在目标 X5 上核对 RDK OS、`uname -m`、GLIBC、二进制 SHA，再执行 `--version`、模型加载、loopback health 和最小 completion；完成并留回执前，`x5_validated`、`model_qualified` 和所有执行/物理权限继续为 `false`。

现有 `install_readonly_llm.py` 仍把该文件当作显式外部、哈希冻结的可执行文件接收，不会复制可执行文件、创建 gate、调用 systemctl 或启动服务。例如：

```bash
python3 rootscope/deploy/x5/scripts/install_readonly_llm.py \
  --release-dir output/rootscope_llm_readonly_release_v1 \
  --project-root rootscope \
  --python /path/to/venv/bin/python3 \
  --llama-server output/rootscope_llama_server_arm64_b9637_v1/bin/llama-server \
  --llama-server-sha256 dcb636215243b8911488b8ca96f0c39bedee14e92f44f7d0ef6c599419acf9b9
```

独立审计：`evidence/rootscope_llama_server_arm64_b9637_independent_audit_20260717.json`。
