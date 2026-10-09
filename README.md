# STATISTICS — 抗屏幕拍摄的统计特征水印

基于**像素域多统计特征与多值量化**的抗屏摄（含部分拍摄/裁剪）鲁棒水印。
核心思想（见 `statistics.txt` 与 `docs/01-overview.md`）：

> 消息不是编码在"某个位置的像素修改"里，而是编码在**水印序列的统计特征**上：
> 消息 → 目标统计量 → 构造满足统计量的 ±1 序列（6 块 × 32 符号）→ U-Net 随机嵌入图像；
> 提取端先恢复序列，再计算 6 个统计特征（比例/均值/方差/偏度/峰度/梯度能量）、
> 8 级量化还原 18 bit 消息。裁剪后幸存的符号是全体符号的无偏子样本，
> 统计量天然稳定 → 抗裁剪。

**规范驱动开发（SSD）**：`docs/` 是唯一事实来源，代码必须与文档一致。
实现参考同工作区的 PIMoG 项目（编码器/解码器/判别器/屏摄噪声层均复用其结构）。

## 目录

```
docs/                 # 规范文档（01 概述 / 02 编码 / 03 架构 / 04 网络 / 05 数据 / 06 评估 / 07 任务）
dataset/              # 你的训练数据（train/ 必需，val/ 可选），自行拷贝
watermark_coding.py   # 消息 ↔ 统计特征 ↔ ±1 序列（核心编码）
noise_layer.py        # 屏摄噪声层（PIMoG 移植）+ 随机裁剪层
data_loader.py        # 数据加载
model.py              # U-Net 编码器 + 残差解码器 + 判别器
solver.py             # 训练 / 验证 / 嵌入 / 提取
main.py               # CLI 入口
smoke_test.py         # 冒烟测试（合成数据，不依赖 dataset/）
tests/test_coding.py  # 编码核心单元测试
```

## 安装

```bash
pip install -r requirements.txt   # torch / torchvision / kornia / numpy / opencv-python
```

## 使用

### 0. 冒烟测试（无需任何数据，验证管线）

```bash
python main.py smoke
```

### 1. 训练（先往 `dataset/train/` 放入图像，详见 `dataset/README.md`）

```bash
python main.py train --dataset_dir dataset/ --num_epoch 100
# 抗裁剪训练：训练时加入随机裁剪（20%~100% 面积后缩放回原尺寸）
python main.py train --dataset_dir dataset/ --num_epoch 100 --crop
```

- 默认屏摄噪声层：`--distortion ScreenShooting`（也可 `Identity` 做干净通道训练）；
- 每 epoch 在验证集上输出：消息准确率（18 bit 整串一致率）、位准确率、序列 BER、PSNR；
- 模型保存至 `checkpoints/`（`best.pth` + 每 epoch 快照），日志在 `logs/`；
- **断点续训**（中断后从上次位置继续，不重头开始）：

```bash
python main.py train --num_epoch 200 --resume checkpoints/checkpoint_epoch_42.pth
```

### 2. 嵌入

```bash
python main.py embed --model_path checkpoints/best.pth \
       --image_dir 你的图片目录/ --message 010101010101010101
```

- `--message` 缺省时逐图随机生成消息；
- 含水印图存到 `results/embed/`，地面真值写入 `results/embed/gt.json`。

### 3. 提取

```bash
python main.py extract --model_path checkpoints/best.pth \
       --image_dir 拍摄图目录/ --watermark_file results/embed/gt.json
```

- 输出每图 18 bit 消息、6 个特征值、6 个量化级别；
- 提供 gt.json 时统计消息准确率、位准确率、序列 BER。

### 4. 真实屏摄测试流程（参考 PIMoG）

1. `embed` 生成含水印图 → 屏幕放大显示并拍摄（占屏 ≥ 1/4）；
2. 用 PIMoG 的 `PerspectiveTransformation/*.m` 做透视校正；
3. `extract` 校正后的拍摄图；
4. 裁剪测试：裁剪拍摄图 20%~80% 后再提取（统计层天然容错）。

## 关键参数（默认值）

| 参数 | 值 | 说明 |
|------|-----|------|
| 图像尺寸 | 256×256 | `--image_size`（128 亦可用；256 下默认 batch=4，4GB 显存可训练） |
| 容量 | 18 bit | K=6 特征 × M=8 级（每特征 3 bit） |
| 序列长度 | 192 | 6 块 × 32 符号（±1） |
| λ1 序列损失 | 3 | 提取序列 MSE |
| λ2 图像损失 | 1 | 梯度掩膜引导 |
| λ3 GAN | 0.001 | 判别器 |
| λ4 统计损失 | 1 | 软统计特征逼近目标 |
| 学习率 | 1e-4 | Adam |

## 文档与代码对照

| 规范 | 实现 |
|------|------|
| `docs/02-coding-spec.md` | `watermark_coding.py` + `tests/test_coding.py` |
| `docs/04-network-spec.md` §4 | `noise_layer.py` |
| `docs/05-dataset-spec.md` | `data_loader.py` |
| `docs/04-network-spec.md` §1~3 | `model.py` |
| `docs/04-network-spec.md` §5~7 | `solver.py` |
| `docs/03-architecture.md` §4 | `main.py`、`config.py` |
| `docs/06-evaluation-spec.md` §4 | `smoke_test.py` |

## 已知限制（v1，详见 `docs/06-evaluation-spec.md` §5）

- 被裁剪位置在提取序列中输出趋近 0，统计量随裁剪面积被"稀释"（`--crop` 训练可部分缓解）；
- N=32/M=8 下单符号翻转可能引起 1 个级别误差（格雷码已把代价压到 1 bit）；
  增大每块符号数（N≥64，L=384）可彻底解决；
- 全量训练与调参需自行执行（冒烟测试仅验证管线）。
