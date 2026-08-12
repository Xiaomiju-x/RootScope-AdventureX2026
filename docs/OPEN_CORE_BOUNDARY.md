# 公开发布边界 / Public Release Boundary

> 历史文件名保留以避免旧链接失效。当前仓库不再是早期的简易 open-core 版本，而是完整、可审计的源码发布。

## “完整开源”是什么意思

仓库发布复现 RootScope 所需的项目自研源码、最终 STM32 工程、RDK X5 运行代码、研究/训练/转换/发布流水线、硬件资料、测试、文档、清理后的证据，以及项目有权再分发的最终模型/媒体资产。

它不是开发硬盘的字节级镜像。以下材料既不增加可复现性，也不能安全/合法公开，因此被明确排除：

| 排除项 | 原因 | 替代方式 |
|---|---|---|
| 密码、API key、token、私钥、SSH/VPS/云凭据 | 安全 | `.env.example` / 占位符；使用者自行配置 |
| 真实 IP、MAC、machine-id、boot-id、序列号、用户绝对路径 | 隐私与设备安全 | 通用配置模板与公开 receipt 清理 |
| 上游基础模型重复副本 | 体积、版本与许可证 | 记录上游修订、下载说明和转换脚本 |
| 无再分发授权的数据/图片 | 版权/肖像/隐私 | 发布 manifest、来源登记与合规获取流水线 |
| 缓存、虚拟环境、优化器状态、临时 checkpoint、编译对象、日志/PID | 可再生且噪声巨大 | requirements、lock/manifest、构建命令与哈希 |
| 原始未脱敏设备/服务器回执 | 凭据与身份泄露 | 清理后的公开证据与技术字段 |
| 原始人物/位置元数据 | 肖像与位置隐私 | 获授权、去 EXIF/GPS、裁切/模糊后的媒体 |

排除这些内容不等于保留技术实现：执行链源码、固件安全状态机和复现步骤均可审阅。安全来自明确合同、默认关闭和硬件工程，而不是“隐藏串口代码”。

## 资产逐项可追溯

- 源码：Apache-2.0，第三方文件保留上游声明；
- 模型：每项有模型卡、上游、许可、大小、SHA-256 和验证边界；
- 数据：每项有来源组、许可、审核和再分发状态；
- 媒体：有处理清单、版权/肖像/商标说明和公开哈希；
- 证据：有 schema、claim、evidence level、输入哈希和清理政策；
- 构建物：内容寻址并可由流水线重建。

## 安全并非秘密性

公开固件和控制代码不能让复现者跳过：独立急停/断能、身份核验、默认关闭、心跳、硬超时、锁存、限位、漏水保护、分级验收和人工监护。网页与 LLM/RAG 不获得物理执行权限。详见 [`SECURITY.md`](../SECURITY.md)。

## English summary

The historical filename is retained for compatibility, but the repository is now a full auditable source release—not the earlier minimal open-core snapshot. Complete means project source, firmware, pipelines, hardware docs, tests, sanitized evidence, and redistributable final assets. It does not mean publishing credentials, device identity, unlicensed data, upstream model duplicates, caches, or unsanitized media. Those exclusions are replaced by templates, manifests, acquisition/build pipelines, hashes, and explicit licensing records.
