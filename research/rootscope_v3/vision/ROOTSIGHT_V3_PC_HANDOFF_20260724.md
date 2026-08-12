# RootSight v3 PC 侧交接

RootSight v3 当前冻结为“双相视觉证据链”：

1. 动作前：CPU ONNX 主判、图像质量门、Energy OOD、split-conformal 集合与
   `unknown` 拒答共同决定 `CLASSIFY / HOLD`。
2. 动作后：固定 ROI 的相对 before/after 变化、目标覆盖、邻区串水与选择性比值
   只提供可见变化证据。
3. BPU r7 以 canonical `hrt_model_exec` 为数值 oracle，以 persistent native
   `libdnn` worker 做最终候选的 43 样本资格回放。Python `hbm_runtime`
   仅保留为非权威 `FAIL_CLOSED` 负路径，不冒充通过后端。
4. 所有视觉输出均为零动作权限；视觉不能打开串口、GPIO 或水泵。

PC `PASS` 仅表示冻结四图 CPU/X5 静态奇偶性与两个湿润选择性夹具通过。现场
USB 摄像头、黄色灯光、固定曝光/白平衡、真实沙舱前后帧、BPU persistent 与
物理灌溉仍须在最终候选激活且硬件到位后单独验收。
