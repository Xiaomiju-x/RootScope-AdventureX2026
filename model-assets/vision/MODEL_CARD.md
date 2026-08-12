# RootSight fixed-card ResNet-18

This 44.7 MB ONNX model classifies exactly four printed AdventureX answer cards:
`grass_clump`, `low_shrub`, `young_tree`, and `non_target`. It is used together with
AKAZE/RANSAC geometric matching; disagreement or poor quality holds action.

The reported same-session holdout was 8/8, which is a fixture result, not natural
plant accuracy and not field generalization. The ONNX model has no physical action
authority. See `model_manifest.json` for preprocessing, thresholds and full rows.
