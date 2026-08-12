# RootScope 四格 A4 彩色打印卡

打印文件：`RootScope_A4_four_up_field_cards_20260723.pdf`

## 位置映射

| A4 横向位置 | 内容 | 裁开后背面标记 |
|---|---|---|
| 左上 | 草丛 `grass_clump` | `G` |
| 右上 | 灌木 `low_shrub` | `S` |
| 左下 | 幼树 `young_tree` | `T` |
| 右下 | 无目标/裸沙 `unknown` | `U - 不得登记为正类模板` |

打印卡正面没有类别文字，避免视觉模型利用文字捷径。裁开后只在卡片**背面**写 `G/S/T/U`。

## 打印与裁切

1. 纸张选择 `A4`，方向选择`横向`。
2. 彩色、单面、高质量或照片质量；推荐哑光白纸或 120-160 g 白色卡纸，避免高光相纸产生反光。
3. 缩放选择`实际大小 / 100%`，关闭`适应页面 / Fit to page`。
4. 先沿中间竖向虚线裁切，再沿中间横向虚线裁切，得到四张约 A6 大小的卡片。
5. 不要再裁掉每张卡片内部的白边；拍摄时保留卡片周围环境，使图像不要占满整个相机画面。

## 现场拍摄建议

- 相机使用稳定路径：`/dev/v4l/by-id/usb-Web_Camera_Web_Camera_202604081837-video-index0`。
- 首轮建议用 `1920x1080 MJPG 30 FPS`；如链路压力大，改用 `1280x720 MJPG 30 FPS`。
- 卡片保持平整，避免屏幕翻拍、镜面反光和严重卷曲。
- 每张卡先拍正视，再拍轻微左/右倾斜和两个距离；卡片主体建议占画面约 35%-70%，不要把原图全幅直接塞满画面。
- `unknown` 只用于拒答验证，不能登记为植物正类模板。
- 打印和复拍不会自动获得训练集、精度、相机或物理闭环资格；现场复拍必须另存为新的 optical-domain 证据。

## 原始图片路径

- 草丛：`datasets/rootscope_machine_curated_provisional_v3/images/grass_clump/grass_clump_163498042_b1f6262895c3.jpg`
- 灌木：`datasets/rootscope_machine_curated_provisional_v3/images/low_shrub/low_shrub_68787114_810c7649ac72.jpg`
- 幼树：`datasets/rootscope_machine_curated_provisional_v3/images/young_tree/young_tree_92774234_0d994e838a2d.jpg`
- 无目标/裸沙：`datasets/rootscope_machine_curated_provisional_v3/images/unknown/unknown_157364276_04e7f49a1e66.jpg`

## 来源与许可

- 草丛：Krzysztof Ziarnek, Kenraiz，Wikimedia Commons，CC BY-SA 4.0。
- 灌木：USDA NRCS Montana，Wikimedia Commons，Public domain。
- 幼树：Wogatha Kanyi，Wikimedia Commons，CC BY-SA 4.0。
- 裸沙：Mostafameraji，Wikimedia Commons，CC BY-SA 4.0。

精确来源页、文件 SHA-256、版面尺寸和真实性边界见相邻的
`RootScope_A4_four_up_field_cards_20260723_manifest.json`。
