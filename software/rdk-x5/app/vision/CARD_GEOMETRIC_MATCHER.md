# RootScope 已知打印卡几何核验

`card_geometric_matcher.py` 只核验一个很窄的事实：查询画面中是否存在与某张**已登记、字节哈希固定的模板卡**相符的局部特征与透视几何。它不是植物语义识别器，不能证明画面中的真实植物类别，也不拥有灌溉、泵、串口或状态机写权限。

## 算法与硬门禁

默认优先 AKAZE，首选通道未通过全部门禁时才尝试 ORB 降级。两个通道都采用二进制描述子 Hamming 距离、`k=2` KNN ratio filter、模板到查询与查询到模板的双向一致匹配，以及 RANSAC homography。最终必须同时通过：

- 模板关键点数与查询关键点数；
- 双向一致 good match 数；
- RANSAC inlier 数和 inlier ratio；
- inlier 中位重投影误差；
- 模板四角投影后的凸性；
- 投影面积占查询画面的范围；
- 投影四边形在查询图像边界内（允许显式像素 margin）。

全部阈值都在 `MatcherConfig` 中，并逐项原样写入 JSON 结果。示例配置是 [`card_geometric_matcher.config.example.json`](card_geometric_matcher.config.example.json)。配置 JSON 含未知字段会直接报错，防止拼写错误被静默忽略。

## PC 命令行

在 `rootscope` 目录运行：

```powershell
$py='python'
& $py -m app.vision.card_geometric_matcher --dump-default-config
& $py -m app.vision.card_geometric_matcher `
  --template C:\path\registered_card.png `
  --query C:\path\camera_frame.png `
  --template-id young-tree-card-v1 `
  --template-class young_tree `
  --config-json app\vision\card_geometric_matcher.config.example.json `
  --output-json C:\path\geometric_result.json
```

退出码 `0` 表示几何门通过，`2` 表示正常核验但门禁拒绝，`1` 表示输入、配置或运行错误。无论退出码为何，输出中的 `irrigation_execution_authority` 和全部 `authority` 字段始终为 `false`。

结果包含模板原文件 SHA-256（数组输入则是 dtype、shape 与连续像素字节的规范摘要）、查询摘要、OpenCV/Python/平台、实际选择的 detector、每次 detector 尝试、完整指标、每个门禁的阈值/值/结论与拒绝原因。

## 与语义分类融合

`fuse_known_card_consensus(...)` 只有在下列三项同时成立时才返回 `KNOWN_CARD_CONSENSUS`：

1. 独立 semantic gate 明确通过；
2. semantic class 与已登记 template class 完全相同；
3. 本几何核验的所有门禁通过，且几何结果绑定的 template class 未被替换。

其他任何情况都是 `REJECT`。即使形成 consensus，也只是“语义证据与已知模板实例几何证据一致”，不产生灌溉执行 authority。

## X5 边界

当前仅在 Windows RTX4050 笔记本的 AdventureX 虚拟环境安装并验证了 `opencv-python-headless`。实际 Windows 二进制摘要见 [`opencv_windows_provenance.json`](opencv_windows_provenance.json)。RDK X5 是 aarch64，**不能复制 Windows wheel 或 `cv2.pyd`**；上板时只先做系统包预检：

```bash
python3 - <<'PY'
import cv2, platform
print(cv2.__version__)
print(platform.machine())
assert platform.machine() in {"aarch64", "arm64"}
assert hasattr(cv2, "AKAZE_create")
assert hasattr(cv2, "ORB_create")
PY
```

该预检说明不代表 X5 依赖已经打包、部署或真机运行。真实模板也尚未锁定；不得用合成单元测试模板冒充赛场模板，不得修改已有训练/验证/打印 holdout 角色。
