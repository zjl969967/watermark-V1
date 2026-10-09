# 04 · 网络与训练规范

> 结构复用 PIMoG（`../PIMoG-.../model.py`、`Noise_Layer.py`、`solver.py`），改动点逐一标注。
> 实现模块：`model.py`、`noise_layer.py`、`solver.py`。

## 1. 编码器 Encoder（`U_Net_Encoder_Diffusion` 改造）

PIMoG 结构不变（3 通道 H×W 输入 → 3 通道输出含水印图；**默认 256×256，128 亦可用**），仅两处改动：

| 项 | PIMoG | 本方案 |
|----|-------|--------|
| 消息长度 | 30 | **192** |
| 消息取值 | {0,1} | **{±1}**（直接作为全连接输入） |

结构要点（照抄 PIMoG 超参）：
- 下采样路径 DoubleConv(3→16→32→64)，MaxPool 逐级减半；
- 消息注入：`Linear(192→256)` → view(1,16,16) → DoubleConv(1→64)，在瓶颈与 3 个
  上采样层级分别插值到目标分辨率并 concat；**瓶颈注入尺寸动态跟随 x4**
  （128 输入 → 16×16；256 输入 → 32×32），因此任意输入分辨率可用；
- GlobalPool(kernel=4) + repeat(4×) 的全局上下文注入保持不变（与 x4 尺寸自动匹配）；
- 输出 `Conv2d(16→3, 1×1)`，无激活（与 PIMoG 一致，输出视为 [−1,1] 附近）。

## 2. 解码器 Decoder（`Decoder`/`Extractor` 改造）

- 前端 `layer1`：3×SingleConv(3→64) + 3×ResidualBlock，与 PIMoG 相同；
- `Extractor`：SingleConv + 3 组 (ResidualBlock×2，stride 2) → 1 通道空间特征 →
  **AdaptiveAvgPool2d(16×16)** → 256 → `Linear(256→192)`（**PIMoG 为 256→30**；
  池化使读出头不依赖输入分辨率：128 输入 16×16 不变，256 输入 32×32→16×16）；
- 输出 192 维实值 v，训练目标为 ±1（MSE），推理时硬阈值（02 文档 §6）。

## 3. 判别器 Discriminator

PIMoG `Discriminator` 原样复用（3×ConvBNRelu(64) + AdaptiveAvgPool + Linear(64→1)），
GAN 损失与交替更新顺序与 PIMoG `solver.train_mask` 相同。

## 4. 噪声层 `noise_layer.py`

### 4.1 移植（PIMoG `Noise_Layer.py` 原样）

- `perspective`（kornia 透视，幅度 d=2）、`Light_Distortion`（线性/径向光照）、
  `Moire_Distortion`（3 通道摩尔纹）、高斯噪声 σ=√0.001；
- `ScreenShooting.forward`：透视 → 光照×0.85 + 摩尔纹×0.15 → 高斯噪声，流程不变；
- `Identity`：恒等。
- 移植时修复 PIMoG 两个已知问题：(a) `Light_Distortion` 中 direction 2/3/4 均等价
  rot90×1（保持原语义，仅注释说明）；(b) 摩尔纹/光照用 numpy 计算后 `torch.from_numpy`，
  保证与 kornia 透视结果设备一致。

### 4.2 新增：随机裁剪层 `RandomCrop`（D6 核心）

- **启用方式**：训练时 `--crop`；默认关闭（关闭时训练仅过屏摄噪声层）。
- **语义**：模拟"部分拍摄"——随机裁剪原图面积的 a∈[0.2, 1.0]（`crop_min_area=0.2`），
  长宽比在 [3/4, 4/3] 内随机，裁剪窗口位置均匀随机，随后**双线性缩放回原尺寸**
  （128 或 256，由输入决定）。
- **实现（可微分）**：kornia `warp_affine`，每样本仿射矩阵 = 以裁剪窗口中心为旋转中心、
  scale=(H/h_c, W/w_c) 的缩放矩阵（`get_affine_matrix2d` + `get_rotation_matrix2d`），
  `padding_mode='border'`；梯度经双线性采样回传。
- 前向签名：`RandomCrop(min_area=0.2)`，`forward(x: BCHW) → BCHW`。
- 推理阶段不使用裁剪层（提取端输入为完整或实拍裁剪后图像）。

## 5. 损失函数（`solver.py`）

| 损失 | 公式 | 权重 | 作用 |
|------|------|------|------|
| 序列损失 L_msg | MSE(v, s)，s 为 ±1 目标 | λ1=3 | 保证序列逐位可提取（PIMoG message loss 同款） |
| 图像损失 L_img | MSE(I_w·mask, I·mask)（见下） | λ2=1 | 视觉不可见性（PIMoG 梯度掩膜版，去掉 v_mask 项） |
| GAN 损失 L_gan | BCE(D(I_w), 真) | λ3=0.001 | 逼真度 |
| 统计损失 L_stat | MSE(norm(soft_stats(v)), norm(T)) | λ4=1 | 引导提取序列统计量逼近目标（02 文档 §7） |

- **梯度掩膜**：mask = (1 − minmax(|∂L_msg/∂I|)) + 1（逐样本归一化），与 PIMoG 相同；
  L_img = MSE(I_w·mask, I·mask)。PIMoG 的 v_mask 通道项因数据无掩膜图而删除。
- **统计损失**：`soft_stats(v, τ=1)` 得 6 维软特征，目标 T 为该图像消息对应的
  6 个网格值；两者均做逐特征 min-max 归一化（min/max 取自 02 文档 §4 网格表）后求 MSE。
  注意：软阈值下软特征与硬 ±1 特征尺度一致（u∈[0,1] 对应比例空间，w=2u−1 对应 ±1 空间），
  归一化只用于平衡 6 特征量纲。
- 总损失：L = λ1·L_msg + λ2·L_img + λ3·L_gan + λ4·L_stat；更新顺序：先判别器、后生成侧
  （与 PIMoG 相同）。

## 6. 训练流程（`solver.py`）

- 优化器：Adam，lr=1e-4（生成侧与判别器同，与 PIMoG 一致）；
- 每 iter：编码 → 噪声 → [裁剪] → 解码 → 损失 → 判别器更新 → 生成侧更新；
- 每 `eval_every` epoch 在 val 集上评估：**消息准确率**（18 bit 整串一致率）与
  **序列 BER**（192 位平均误码率），日志写入 `logs/train_<distortion>[_crop].txt`；
- 保存策略：每 `model_save_step` epoch 存 `checkpoints/checkpoint_epoch_<e>.pth`，
  另存 `checkpoints/best.pth`（按 val 消息准确率）；
- checkpoint 内容：`{'encoder','decoder','discriminator','optimizer','optimizer_D',
  'epoch','best_acc'}`（含优化器状态，保证续训无缝）；
- **断点续训**：`--resume checkpoints/checkpoint_epoch_<e>.pth`——恢复模型/判别器/
  优化器/epoch/best_acc，从 epoch e+1 继续；日志以追加模式续写，best.pth 只在新结果
  更好时更新（不会被重启的差结果覆盖）；
  （单 GPU，不使用 DataParallel，与 PIMoG 的 DataParallel 写法不同——规范以此为 v1 约定）。

## 7. 评估 / 嵌入 / 提取（`solver.py`）

- `embed`：Encoder 前向（不接噪声层），输出反归一化到 [0,255] 存 PNG；
  记录每图 PSNR（对原图）与残差幅度；
- `extract`：Decoder 前向 → 硬阈值 → `sequence_to_message`；记录 6 特征值、6 级别、
  序列 BER（若有真值）、消息准确率；
- 真实屏摄测试流程见 `06-evaluation-spec.md` §2（透视校正沿用 PIMoG 的 MATLAB 脚本，
  不在本 v1 代码范围内）。

## 8. 设备与性能

- 自动选择 CUDA/CPU；混合精度不启用（v1 保持简单）；
- 冒烟测试在 CPU 也可完成（批 2、合成 32 图）；全量训练建议 GPU。
