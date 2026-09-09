# 03 · 架构规范（目录 / 模块 / CLI）

## 1. 目录结构

```
statistics/
├── docs/                    # 规范文档（SSD 事实来源）
│   ├── 01-overview.md
│   ├── 02-coding-spec.md
│   ├── 03-architecture.md
│   ├── 04-network-spec.md
│   ├── 05-dataset-spec.md
│   ├── 06-evaluation-spec.md
│   └── 07-task-breakdown.md
├── dataset/                 # 用户自备数据（空目录，含 README 说明）
│   ├── train/               # 训练图像（*.jpg / *.png / *.bmp）
│   └── val/                 # 验证图像（可缺省 → 自动 90/10 拆分）
├── smoke_data/              # 冒烟测试合成数据（由 smoke_test.py 生成，勿手改）
├── checkpoints/             # 模型权重（train 输出）
├── logs/                    # 训练日志 txt
├── results/                 # 嵌入图像 / 测试输出
├── main.py                  # CLI 入口（train / embed / extract / smoke）
├── config.py                # 配置 dataclass + argparse 构建
├── watermark_coding.py      # 编码核心（02 文档）
├── noise_layer.py           # 屏摄噪声层 + 随机裁剪层（04 文档 §3）
├── data_loader.py           # 数据加载（05 文档）
├── model.py                 # 编码器/解码器/判别器（04 文档 §1~2）
├── solver.py                # 训练循环 / 验证 / 嵌入 / 提取（04 文档 §6）
├── smoke_test.py            # 冒烟测试（06 文档 §4）
├── tests/
│   └── test_coding.py       # 编码单元测试（02 文档 §9）
├── requirements.txt
└── README.md
```

## 2. 模块依赖（自底向上，实现顺序即本顺序）

```
watermark_coding ──(无依赖，纯 numpy/torch 张量操作)
noise_layer      ──(仅 torch + kornia)
data_loader      ──→ watermark_coding（训练时生成随机消息序列）
model            ──→ noise_layer（ScreenShooting / Identity 接线）
solver           ──→ model, data_loader, watermark_coding, config
main             ──→ config, solver, smoke_test
smoke_test       ──→ 以上全部 + 合成数据生成
```

依赖关系决定任务拆解顺序，见 `07-task-breakdown.md`。

## 3. 配置（`config.py`）

单一 dataclass `Config`，由 `build_arg_parser()` 生成 argparse 参数。关键字段：

| 字段 | 默认 | 说明 |
|------|------|------|
| image_size | 128 | 输入/输出分辨率 |
| batch_size | 16 | 批大小 |
| num_epoch | 100 | 训练轮数（全量训练由用户执行） |
| lr | 1e-4 | 编码器/解码器与判别器共用学习率（Adam） |
| lambda1 | 3 | 序列 MSE 损失权重（message） |
| lambda2 | 1 | 梯度掩膜图像损失权重（image） |
| lambda3 | 0.001 | 判别器 GAN 损失权重 |
| lambda4 | 1 | 软统计特征损失权重（stat） |
| distortion | ScreenShooting | 训练噪声：`Identity` / `ScreenShooting` |
| crop | False | 训练时是否启用随机裁剪层（**D6**：`--crop` 开启，默认关） |
| crop_min_area | 0.2 | 裁剪保留面积下限（20%，对应报告"覆盖 20% 以上"） |
| dataset_dir | dataset/ | 数据根目录（其下 train/、val/） |
| checkpoint_dir | checkpoints/ | 模型保存目录 |
| log_dir | logs/ | 日志目录 |
| result_dir | results/ | 输出目录 |
| message | 随机 | 嵌入模式使用的 18 bit 消息字符串 |
| watermark_file | 无 | 提取模式的地面真值 JSON（可选） |
| model_path | 无 | 嵌入/提取模式加载的 checkpoint |
| num_workers | 0 | DataLoader 工作进程数（Windows 下默认 0） |
| log_step | 40 | 训练日志间隔（iter） |
| eval_every | 1 | 验证频率（epoch） |
| tau | 1 | 软统计温度（04 文档 §5） |

## 4. CLI（`main.py`）

```
python main.py train   [--dataset_dir dataset/] [--num_epoch 100] [--crop] [--distortion ScreenShooting|Identity] ...
python main.py embed   --model_path checkpoints/xxx.pth [--image_dir results/embed_in/] [--message 010101...] 
python main.py extract --model_path checkpoints/xxx.pth [--image_dir ...] [--watermark_file gt.json]
python main.py smoke   [--smoke_iters 5]
```

- `--crop`：仅 `train` 子命令接受；**默认不加裁剪**（D6），添加后训练数据流为
  编码 → 屏摄噪声 → 随机裁剪 → 解码。
- `embed`：对目录内每张图嵌入指定（或随机）消息，保存含水印图到 `results/embed/`，
  同时写出地面真值 `results/embed/gt.json`（消息与文件名映射）。
- `extract`：对目录内每张图输出提取消息与逐张准确率；若提供 `watermark_file` 则对比统计
  总体消息准确率（18 bit 级）与序列 BER（192 位级）。

## 5. 训练数据流（每 iter）

```
batch 图像 I (B,3,128,128) ∈ [−1,1]
随机消息 m_b (18bit) → 构造序列 s_b (±1,192)
Encoder(I, s) → I_w
NoiseLayer(I_w) → I_n          # ScreenShooting 或 Identity
[若 --crop] Crop(I_n) → I_n
Decoder(I_n) → v (B,192)
损失 = λ1·MSE(v, s) + λ2·图像损失(梯度掩膜) + λ3·GAN + λ4·统计损失(soft)
更新：判别器 D（先）→ 编码器+解码器（后），与 PIMoG 相同交替顺序
```

## 6. 嵌入 / 提取数据流

- **嵌入**：I → Encoder(I, s) → I_w → 存盘（不经过噪声层）。
- **提取**：I_n → Decoder → v → 硬阈值 → `sequence_to_message` → m̂；
  中间量（6 特征值、6 级别、序列 BER）全部记录，便于诊断。

## 7. 复现性

- 构造序列：`rng=np.random.default_rng(0)`（同一消息恒得同一序列）；
- 训练/数据加载 RNG：由 `--seed`（默认 0）控制 numpy/torch/DataLoader；
- 噪声层内随机量（透视/光照/摩尔纹）沿用 PIMoG 的每次前向独立随机，不设种子
  （这是数据增强的一部分，不影响推理确定性）。
