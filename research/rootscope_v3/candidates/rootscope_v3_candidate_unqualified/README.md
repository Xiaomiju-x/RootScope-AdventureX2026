# RootScope v3 candidate 骨架

状态：`SKELETON_NOT_A_RELEASE_DO_NOT_DEPLOY`

此目录仅定义未来候选的布局和 fail-closed 状态。E0 没有向其中放入
模型、运行 payload 或 X5 资格回执。只有以下条件全部完成后，后续构建器
才能创建新的、名称不同的不可变 release：

1. 模型/数据/依赖逐项哈希绑定；
2. 五套冻结评测合同均生成真实结果；
3. PC 静态门通过；
4. X5 上电后的 CPU、BPU、相机、LLM 和 combined soak 门通过；
5. 零权限边界和回滚门通过；
6. 物理闭环未完成时明确保持 `physical_completion=false`。

禁止把本骨架原位改名为 release。正式候选必须由构建器输出到新的
`output/releases/<immutable_candidate_id>/`。
