# 复现与交付说明

本项目输入 RGB 图像和英文抓取指令，输出原图坐标系中的二维抓取框 `(x,y,w,h,theta)`。角度为顺时针弧度，周期为 π。推理不读取目标 mask 或 GT；mask 仅用于离线构建训练配对和检查标注。

## 提交版实验

正式记录为 `runs/submission_001`。训练/验证/测试分别包含 7,477/704/605 张图像、25,701/2,486/1,960 条指令，验证和测试与训练按图像分离。3,062/328/189 个有效配对分别来自 386/43/36 张图。362 条单位不明确的可疑几何记录被隔离。

每种方法各训练 seed 42、123、3407，5,050 步，BF16、batch 256，其中 128 条指令来自 64 个配对。Baseline 与 SwapGrasp 在同种子下使用完全相同的样本顺序和初始化，区别是 `lambda_swap=0/0.1`。验证 Top-1 选择最优 checkpoint；测试不参与选模型。学习率采用相同的验证 plateau 规则，可能因验证轨迹不同而产生不同调度。正式 run 的完整配置在各自 `config.yaml` 中，优先于示例配置。

这是可用标注构成的固定诊断子集，不能与官方全量论文结果直接排名。完整归档通过 SHA-256、提取的图像和标注通过 ZIP CRC；几何检查不保证语言和目标的语义标注无误。

## 环境

在服务器使用已有 `.venv`。新环境应先安装适配 GPU 的 PyTorch/torchvision，再安装本包；原服务器记录为 Python 3.12、PyTorch 2.7.1+cu126、torchvision 0.22.1+cu126，完整依赖在 `artifacts/requirements-resolved.txt`。

```bash
cd /root/labron/swapgrasp
source .venv/bin/activate
pip install -e '.[test]'
python -m pytest -q
```

## 数据与预训练权重

数据需自行取得官方授权，来源为 `airvlab/Grasp-Anything` 与 `airvlab/Grasp-Anything-pp`。下载脚本支持 Hugging Face 镜像传输，记录仓库 revision、文件 LFS SHA-256 和下载校验凭据。

```bash
python download_data.py --workers 4
git clone https://github.com/andvg3/LGD.git third_party/LGD
git -C third_party/LGD checkout 501904dc5c30c8fad58d90f7e4f49b4095f11eb0
python fetch_weights.py
python -c "from pathlib import Path; from torch.hub import download_url_to_file; Path('weights').mkdir(exist_ok=True); download_url_to_file('https://download.pytorch.org/models/resnet18-f37072fd.pth', 'weights/resnet18-f37072fd.pth', hash_prefix='f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec')"
python prepare_real.py --output data/submission_001 \
  --train-images 10579 --val-images 1000 --test-images 620 \
  --verified-local-images --quarantine-subpixel
python tools.py audit --manifest data/submission_001/manifest.jsonl
```

LGD 仅提供官方划分清单和格式参照，本项目模型由 `swapgrasp/model.py` 独立实现。数据准备时 `--seed` 默认 42；从正式结果中的 `provenance/manifest.jsonl` 可以核对最终查询集合。ZIP 中的完整数据与预训练权重未随代码包再分发。

本次 CLIP 权重固定来源为 `timm/vit_base_patch32_clip_224.openai`，revision 为 `a6f597a30f7b82c51704746581f9a4e41421e878`，文件 `open_clip_pytorch_model.bin`，SHA-256 为 `9ecdaef325b20e7283dc6a32f92aa638d100899e4f084c2462d3832eeea0b26e`。若重新下载时上游主分支已变化，应使用本次归档凭据的固定 revision，而不是混用新的权重。ResNet-18 校验值见上方命令。

## 新建训练，勿覆盖已完成目录

```bash
python scripts/run_submission.py \
  --manifest data/submission_001/manifest.jsonl \
  --pairs data/submission_001/train_pairs.jsonl \
  --output runs/reproduction_001 --batch-size 256 --epochs 50 \
  --seeds 42 123 3407 --gpus 0 1 2 3 4 5
python scripts/finish_submission.py --run runs/reproduction_001 --eval-gpu 6
```

第一个命令启动六个独立训练进程后返回，第二个命令负责等待训练、选择已保存的验证最优权重、测试、配对比较与报告。长期运行时在 `tmux`/`screen` 内执行；远端现有 `submission_001` 已独立运行，无需重复执行。准确恢复使用相同配置和 `--resume <run>/last.pt`，不要改变总步数后声称是精确续训。

```bash
python train.py --config <run>/config.yaml --output <run> --resume <run>/last.pt
tensorboard --logdir runs/submission_001 --host 127.0.0.1 --port 6006
```

VS Code 转发服务器 6006 端口即可查看 TensorBoard。记录包括总 loss、中心质量、几何损失、原始/加权 swap loss、梯度范数、学习率、吞吐和显存。`live_training.png` 与各 run 的 `curves.png` 可直接打开。

也可以在本机解压 `results.zip`，执行 `tensorboard --logdir <解压目录> --host 127.0.0.1 --port 6006`，对比六组已保存的训练记录，无需连接服务器。

## 单独推理与评测

任意新图片可以直接运行以下命令，不需要 GT、mask 或手写 manifest：

```bash
CUDA_VISIBLE_DEVICES=6 python infer.py \
  --checkpoint runs/submission_001/swap_s42/best.pt \
  --image /absolute/path/to/image.jpg --text "grasp the blue bottle" \
  --output artifacts/my_prediction
```

输出 `predictions.jsonl`、预测框 PNG、固定色标质量热图与 `index.html`。JSON 中 `grasp` 为原图像素坐标下的 `[x,y,w,h,theta]`，`theta` 为顺时针弧度、周期 π；`score` 用于候选排序，不是经过校准的机器人执行成功概率。

```bash
CUDA_VISIBLE_DEVICES=6 python evaluate.py predict \
  --checkpoint runs/submission_001/swap_s42/best.pt \
  --manifest data/submission_001/manifest.jsonl --split test \
  --output runs/submission_001/recheck_predictions.jsonl
python evaluate.py score --manifest data/submission_001/manifest.jsonl --split test \
  --pairs data/submission_001/train_pairs.jsonl \
  --predictions runs/submission_001/recheck_predictions.jsonl \
  --output runs/submission_001/recheck_metrics
```

标准成功要求同一个 GT 同时满足 IoU > 0.25 且角差 < 30°。严格阈值为 (0.50,15°)，更严格为 (0.75,10°)。Top-5 表示五个候选中存在一个成功候选，是候选集合上界，不等于执行五次。补充结果包括图像宏平均、seen/unseen、几何误差、成对同时成功和同图竞争目标命中率。连续旋转多边形 IoU 尚未与官方光栅化 evaluator 逐样本对齐。

同图目标特异性是训练启动后增加的补充诊断，复用固定最终预测；它不参与选择 checkpoint 或调整训练。差值为正确目标命中率减去竞争目标命中率，按有效配对的两个方向计数；同一查询可能在多个配对中出现。bootstrap 按整张图重采样，保留图内相关性。

## 结果入口

| 文件 | 用途 |
|---|---|
| `runs/submission_001/REPORT.zh-CN.html` | 指标、曲线、解释图一页查看 |
| `runs/submission_001/evidence.json` | 三种子汇总、严格指标、bootstrap 和来源 |
| `runs/submission_001/*/test_predictions.jsonl` | 全测试预测，支持重新评分 |
| `runs/submission_001/*/test/detailed_per_query.jsonl` | 每条查询的成败与几何误差 |
| `runs/submission_001/*/best.pt` | 验证最优权重，保留在服务器 |
| `runs/submission_001/explanations/` | 同图两指令 × 两模型与失败统计 |
| `paper/paper.pdf` | 英文 CVPR 格式短稿，含图、表、公式和参考文献 |
| `artifacts/submission_001/source.zip` | 代码、测试、配置与论文源文件 |
| `artifacts/submission_001/results.zip` | 报告、预测、图表与数据来源清单 |

## 交付边界

原题只要求向 HR 提交英文 PDF，最多两页（含参考文献），文件名为 `FirstName_LastName.pdf`。论文作者姓名与最终 GitHub 仓库 URL 需要由本人填写，并由本人最终审阅。现阶段不向 HR 发送文件；没有虚构作者信息或公开发布数据。源包不包含训练数据、预训练权重和训练 checkpoint，模型权重保存在服务器，可单独提取。
