# X5 CPython 3.10/aarch64 候选 wheelhouse

本目录公开 11 个依赖的精确文件名、版本、大小和 SHA-256 合同，但不把第三方 wheel 二进制重复提交到 Git。它解决“怎样取得同一组离线依赖并验证内容”的问题，不代表这组依赖已经在任意 RDK OS 镜像上通过资格。

当前状态始终是：

```text
CROSS_DOWNLOADED_HASH_LOCKED_NOT_EXACT_TWIN_X5_QUALIFIED
```

Pillow 已在 2026-08-13 从存在公开中危公告的 11.3.0 更新至 12.2.0。新 aarch64 wheel 来自 PyPI，文件名、大小和 SHA-256 已写入清单；这次 PC 侧安全刷新不构成新的 X5 真机验证。

## 1. 在联网 PC 上取得精确候选 wheel

以下命令只下载 Linux aarch64 / CPython 3.10 的公开 wheel，不安装包、不连接设备：

```bash
python -m pip download \
  --only-binary=:all: \
  --platform manylinux2014_aarch64 \
  --platform manylinux_2_27_aarch64 \
  --implementation cp \
  --python-version 310 \
  --abi cp310 \
  --dest deploy/x5/wheelhouse/candidate_cp310_aarch64 \
  --require-hashes \
  -r deploy/x5/wheelhouse/requirements-cp310-aarch64-candidate.txt
```

下载完成后执行严格文件审计：

```bash
python deploy/x5/wheelhouse/audit_candidate_wheelhouse.py \
  --manifest deploy/x5/wheelhouse/candidate_cp310_aarch64_manifest.json \
  --require-wheel-files
```

不带 `--require-wheel-files` 时只验证公开清单自身；安装器会强制使用严格模式，因此缺少、增加或篡改任何 wheel 都会在创建 venv 前失败。

## 2. 在目标 X5 上离线安装候选环境

把整个 `wheelhouse/` 目录安全复制到目标板的隔离 staging 后运行：

```bash
bash deploy/x5/scripts/install_cpu_venv_candidate.sh
```

安装器先强制 Linux/aarch64/CPython 3.10，随后校验全部文件，并使用 `pip --no-index --require-hashes`。它不会联网下载、运行 `apt`、访问相机/串口/GPIO 或修改网络。

安装成功也只产生“候选依赖环境”证据。必须在同一块目标 X5、同一系统镜像中另外完成 import、golden tensor、ONNX CPU self-test 和冷启动回执，才能提出板端资格结论。

禁止把 Windows/x86_64 wheel 放入该目录，也禁止把 PC 交叉下载成功描述成 X5 真机运行成功。
