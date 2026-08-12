# RootScope clean-X5 离线部署胶囊

当前状态：`SIMULATED_ONLY_CLEAN_X5_CAPSULE_NOT_X5_QUALIFIED`。

这个目录解决“拿到一块刚装好系统的 RDK X5 后，如何用同一套离线文件做可复核启动”的软件骨架问题。它现在不是可直接上板的不可变胶囊：exact-twin 板卡、RDK OS 镜像、候选 aarch64 wheelhouse 的真板安装回执、摄像头型号和 BPU runtime 都还没有冻结。当前 CPython 3.10/aarch64 wheel 候选已经逐文件锁哈希，但仍不是 exact-twin 资格证明。

## 当前能做什么

- 严格解析零权限 capsule 配置；任何 authority、BPU-ready、model-candidate 或 physical-completion 为 `true` 都拒绝启动。
- 只读检查项目/Python/依赖/模型/可选设备**显式别名**，不扫描 `/dev`，不打开设备。
- 对可选 ONNX 强制 `CPUExecutionProvider`，校验模型 SHA-256、静态输入/输出、四类顺序和预处理合同，再用确定性模拟 RGB 做一次有限值回放。预处理严格为短边 256 → 中心裁剪 224 → RGB tensor → ImageNet normalize，并用 golden tensor hash 防漂移。
- 启动仅有 GET 状态接口的锁定 Dashboard；没有动作 endpoint。
- 已把 397,805,120 字节的 XRD Qwen2 0.5B GGUF 机械复制到 `adventurex/output/rootscope_llm_readonly_release_v1` 并锁定 manifest/SHA256SUMS；llama-server 仍是外部依赖，未随包提供。
- 给出独立的用户级 llama-server 模板：无硬编码账号、无 `[Install]` 自动启用、默认没有 gate/ack、只监听 `127.0.0.1`，并用 systemd `IPAddressDeny=any` + `IPAddressAllow=localhost` 限制网络。锁定 Dashboard 不查询该服务。

## RGB、深度与 BPU 边界

- `inputs.rgb` 可在未来改成 `uvc_v4l2 + /dev/rootscope_uvc`；当前锁定服务仍不会打开它。
- `inputs.depth` 可声明 `depth_v4l2` 或 `vendor_sdk + /dev/rootscope_depth`；深度只作为未来测距/湿润辅助，不能无合同地喂入当前 RGB 分类器。
- 本目录不导入 `hobot_dnn` 或 `hbm_runtime`，不包含 `.bin`，也不把现有 ONNX 写成 BPU-ready。

## 目标板冻结后的本地流程

以下命令只应在已登记的 exact-twin X5 本机执行；本目录不提供 SSH、网络修改或在线下载脚本。

```bash
# 1. 把已审核的项目 release 放到 /opt/rootscope/current
# 2. 用已冻结 aarch64 wheelhouse 创建 /opt/rootscope/venv
# 3. 复制并填写 /etc/rootscope/capsule_config.json 与 rootscope.env
bash /opt/rootscope/current/deploy/x5/scripts/preflight.sh
bash /opt/rootscope/current/deploy/x5/scripts/start_rootscope.sh
```

## 可选本地 LLM：安装不等于启用

PC 侧可重复 staging（只读 XRD 源，只写 AdventureX/output）：

```bash
python3 rootscope/deploy/x5/scripts/stage_readonly_llm.py
```

X5 上的 `install_readonly_llm.py` 必须显式接收一个已冻结 SHA-256 的 aarch64 `llama-server`；脚本不会复制该可执行文件、不会创建 gate、不会调用 systemctl、不会启动服务。它安装用户级 unit，因此不假设 `rootscope` 账号存在。安装后仍需人工把 env 的 `ROOTSCOPE_LLM_MANUAL_ACK` 改为 `READ_ONLY_EXPLANATION_ONLY`，并创建内容严格相同的 gate 文件，服务才可能启动。任何启用都只能在 exact-twin 本地资格检查之后进行。

显式讲解入口是 stdout-only：

```bash
python3 /opt/rootscope/current/deploy/x5/scripts/explain_readonly_snapshot.py \
  --runtime-config "$HOME/.config/rootscope/rootscope-llm-runtime.json" \
  --snapshot-json /path/to/structured_snapshot.json
```

它先校验 GGUF、release manifest、外部 llama-server 哈希和 loopback `/health`；不会写状态机或调用任何动作接口。

只有 preflight、CPU 模拟回放、锁定页面、重启和回滚均在 exact-twin X5 留下哈希回执后，才能把 `x5_validated` 作为另一个受控 release 的事实；本 v1 合同本身永远保持 false。

## 文件

- `XRD_REUSE_INVENTORY.md`：只复用工程方法的边界盘点。
- `capsule_config.example.json`：默认 model/RGB/depth/LLM 全禁用的零权限示例。
- `offline_dependencies.json`：候选依赖与未冻结项。
- `requirements-*.txt`、`wheelhouse/`：后续 exact-twin 离线包入口，不是当前 lock。
- `scripts/`：纯本机预检与锁定启动。
- `readonly_llm_release_spec.json`：冻结源、目标、大小、SHA 与全 false 正式标志。
- `scripts/stage_readonly_llm.py`：AdventureX 内确定性 staging。
- `scripts/install_readonly_llm.py`、`readonly_llm_preflight.py`、`start_readonly_llm.sh`：用户级禁用安装、哈希预检和手动启动门。
- `systemd/`：主服务和无账号硬编码、无自动 enable 的只读 LLM 用户模板。
- `models/`：已机械复制、仍未资格化的 seed17 CPU 实验 ONNX。
- `capsule_config.seed17_cpu_experimental.json`、`seed17_cpu_deployment_manifest.json`：绑定模型 SHA、接口、类别顺序和训练一致预处理；所有 X5/BPU/物理权限仍为 false。
