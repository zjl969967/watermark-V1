# dataset/ 数据目录说明

本目录由你自行拷贝数据（规范见 `docs/05-dataset-spec.md`）。

## 结构

```
dataset/
├── train/   # 训练图像（必需）：*.jpg / *.png / *.bmp / *.jpeg
└── val/     # 验证图像（可选）：缺省时自动从 train 按 9:1 拆分
```

## 要求

- 普通整幅自然图像即可（**不需要** PIMoG 那种"左图右掩膜"双列格式）；
- 推荐 ≥ 200 张；图像会被统一 resize 到 128×128；
- 放入图像后运行：

```bash
python main.py train --dataset_dir dataset/ --num_epoch 100
# 需要抗裁剪训练时加 --crop
python main.py train --dataset_dir dataset/ --num_epoch 100 --crop
```
