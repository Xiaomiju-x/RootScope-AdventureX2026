# X5 CPython 3.10/aarch64 候选 wheelhouse

`candidate_cp310_aarch64/` 已放入 11 个公开二进制 wheel，包括 CPU ONNX 与打印卡几何支路需要的 NumPy、Pillow、ONNX Runtime、OpenCV 及传递依赖。它们是从 PC 用 `--platform manylinux2014_aarch64` / `manylinux_2_27_aarch64` 交叉下载的，全部文件名、大小和 SHA-256 已锁定。

这解决的是“现场无网时有没有目标架构安装文件”，**没有**解决 exact-twin RDK OS 兼容性：当前没有在比赛提供的 X5 镜像上安装、import、运行 golden preprocess 或回放 ONNX。因此状态仍是 `CROSS_DOWNLOADED_HASH_LOCKED_NOT_EXACT_TWIN_X5_QUALIFIED`。

本地审计：

```bash
python3 deploy/x5/wheelhouse/audit_candidate_wheelhouse.py \
  --manifest deploy/x5/wheelhouse/candidate_cp310_aarch64_manifest.json
```

目标 X5 离线候选安装：

```bash
bash deploy/x5/scripts/install_cpu_venv_candidate.sh
```

安装器先强制 Linux/aarch64/CPython 3.10，再运行 wheel 审计，并使用 `pip --no-index --require-hashes`。它不会下载、运行 `apt` 或修改网络。安装成功也只产生候选依赖证据；还必须在同一块 X5 上保存 import、golden tensor、真实 ONNX CPU selftest 和冷启动回执，才能讨论 X5 qualification。

禁止把 Windows/x86_64 wheel 复制进本目录，禁止把 PC 交叉下载成功写成真机运行成功。
