# 07 · 任务拆解与依赖

> 依赖顺序即实现顺序：上一任务完成（含自测）后进入下一任务。
> 状态：☐ 待开始 / ◐ 进行中 / ☑ 完成。

| # | 任务 | 依赖 | 交付物 | 验收（参照规范） | 状态 |
|---|------|------|--------|------------------|------|
| T1 | 规范文档 | — | `docs/01~07` | 用户已确认决策 D1~D7 全部落档 | ☑ |
| T2 | 项目骨架 | T1 | 目录、`dataset/`(空)+说明、`requirements.txt`、`config.py` | `python -c "import config"` 通过；CLI 参数与 03§3 一致 | ☑ |
| T3 | 水印编码核心 | T1 | `watermark_coding.py` + `tests/test_coding.py` | 02§9 三条单测全过 | ☑ |
| T4 | 噪声层 | T1 | `noise_layer.py`（ScreenShooting/Identity/RandomCrop） | 前向形状不变；裁剪层面积/位置统计符合 04§4.2 | ☑ |
| T5 | 数据加载 | T2,T3 | `data_loader.py` | 05§2 归一化/返回结构正确；空目录报错信息友好 | ☑ |
| T6 | 网络模型 | T4 | `model.py`（Encoder/Decoder/Discriminator） | 前向形状 (2,3,128,128)+(2,192)→(2,192)；参数量合理 | ☑ |
| T7 | 训练/评估/嵌入/提取 | T2,T5,T6 | `solver.py` | 04§5~7 四损失齐全；checkpoint 结构符合约定 | ☑ |
| T8 | CLI | T7,T3 | `main.py` | `train/embed/extract/smoke` 四子命令；`--crop` 仅 train 生效 | ☑ |
| T9 | 冒烟测试 | T2~T8 | `smoke_test.py` + `smoke_data/` | 06§4 S1~S6 全过，`SMOKE TEST PASSED`（退出码 0） | ☑ |
| T10 | README | T1~T9 | `README.md` | 安装、四命令示例、与规范文档对照、已知限制 | ☑ |

## 依赖图

```
T1 规范文档
 └─ T2 骨架 ── T3 编码核心 ──┐
    ├─ T4 噪声层 ── T6 网络 ─┤
    └─ T5 数据加载 ──────────┤
                             └─ T7 solver ─ T8 CLI ─ T9 冒烟 ─ T10 README
```

## 执行约定

1. 每任务完成后运行对应自测再进入下一任务；
2. 实现与规范冲突时回改规范并记录（保持 SSD 一致性）；
3. 需要用户输入/数据时停下提问；冒烟测试使用合成数据，不阻塞用户数据准备。
