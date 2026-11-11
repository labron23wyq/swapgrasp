# SwapGrasp

## 已完成的提交版结果（2026-11-11）

六组训练及全部验证/测试评分已完成，每组 5,050 步。三种子均值：Baseline / SwapGrasp 标准 Top-1 为 **53.18% / 52.64%**，Top-5 为 **82.19% / 82.70%**，严格 Top-1 为 **45.31% / 43.62%**，同图成对 Top-1 为 **35.80% / 31.04%**。所有种子的标准 Top-1 差值图像聚类 bootstrap 95% 区间均包含零；本实验未建立 swap loss 的稳定收益。完整逐种子结果、严格阈值与失败图见 `runs/submission_001/REPORT.zh-CN.html`，两页英文论文见 [`paper_revision_001/paper.pdf`](paper_revision_001/paper.pdf)，LaTeX 源码见 [`paper_revision_001/paper.tex`](paper_revision_001/paper.tex)。这是自定义图像隔离子集结果。

单图推理使用 `python infer.py --checkpoint <best.pt> --image <image.jpg> --text "grasp the blue bottle" --output artifacts/my_prediction`，不需要标注。已验证的代码和结果包位于 `artifacts/submission_001/`；权重保留于 `runs/submission_001/*/best.pt`。作者为 Yuquan Wang，公开源码仓库地址为 https://github.com/labron23wyq/swapgrasp。

## 提交版三种子实验（2026-11-10 启动）

复现步骤、指标定义与可交付文件清单见 [`REPRODUCE.zh-CN.md`](REPRODUCE.zh-CN.md)。这份 README 顶部记录最新主实验，其余早期 pilot 与工程背景为历史说明。

服务器 `/root/labron/swapgrasp/runs/submission_001` 已配置六个独立 run：GPU 0/1 为 seed 42 的 baseline/swap，GPU 2/3 为 seed 123，GPU 4/5 为 seed 3407。使用完整 SHA-256 校验的本地图像与标注归档，固定官方 object 列表所覆盖的图像候选，移除 train/test 共享图像后再划分验证集。实际可用数据为 train/val/test 7,477/704/605 张图、25,701/2,486/1,960 条指令；362 条可疑亚像素几何记录隔离在 `data/submission_001/quarantine.jsonl`。这不是百万图全量 benchmark。

batch size 经真实数据显存/吞吐测试选为 **256 条指令，每步 64 对**；每组 5,050 步，共约 50 个 query 等效 epoch。配对采样占一半 query，等效 epoch 不代表均匀遍历数据。基线和方法的采样、初始化种子及配置相同，仅 swap 权重为 0 / 0.1。最优权重按验证 Top-1 选择，测试不参与调参。

每个 run 的 `tensorboard/` 保存总 loss、quality、geometry、原始及加权 swap loss、学习率、梯度范数、吞吐和显存峰值。`curves.png` 与总目录 `live_training.png` 自动刷新。TensorBoard 在服务器 `127.0.0.1:6006`，VS Code Ports 转发 6006 后可访问。

运行完成后，`scripts/finish_submission.py` 自动评测全部验证/测试指令、生成固定示例可视化、计算三种子对照区间和报告。`scripts/postprocess_submission.py` 更新英文短稿并生成 `artifacts/submission_001/source.zip` 和 `results.zip`。状态分别见 `workflow_state.json` 与 `postprocess_state.json`；状态为 running 时不能把训练计划当作完成结果。

```bash
python scripts/monitor_submission.py runs/submission_001
python scripts/report_submission.py --run runs/submission_001
```

评分保留 IoU>0.25 / angle<30° 的原始 Top-1/5，同时新增 IoU>0.50 / angle<15° 与 IoU>0.75 / angle<10°，以及多阈值网格、同图配对成功、按图像宏平均、seen/unseen 和 Top-1 单一 GT 对应的几何误差分布。`evaluate.py score` 同时写出 `metrics.json`、`detailed_metrics.json`、`per_query.jsonl` 和 `detailed_per_query.jsonl`。严格阈值是补充诊断定义；连续多边形 evaluator 尚未与官方光栅化版本逐样本对齐。

语言条件二维抓取检测：RGB + 英文指令 → `(x,y,w,h,theta)`。本项目独立实现 ResNet-18 / FPN / FiLM / 密集抓取头，使用冻结 CLIP 文本塔；在相同 baseline 上增加同图互斥目标的双向排序损失。

## 当前状态

以下首轮 pilot 和开发记录不代表 2026-11-10 启动的提交版六组训练已完成；该实验实时状态以 `runs/submission_001/workflow_state.json` 为准。

- 模型、稀疏几何监督、双向 swap 损失、同图共享视觉计算、训练与验证、完整 checkpoint 恢复、独立推理/评分和可视化已实现。
- 工程验证使用明确标记的合成样本。`debug_hash` 没有语言语义能力，代码禁止用它训练或评测真实数据。
- 所用标注包和本次提交版使用的完整图像归档已完成全档 SHA 校验；提取成员经 CRC 检查。较早 pilot 曾使用按成员 CRC 获取的图像前缀。
- 已完成 85 张图 / 399 条指令的首轮真实数据诊断实验：训练 53 张、验证 8 张、测试 24 张；baseline 和 swap 各训练 1,500 步。测试 Top-1 为 19.2% / 32.7%，但图像聚类 bootstrap 的 95% 区间触及 0，不能宣称稳定提升。测试配对同时成功率均为 0%，尚需解决多目标语言泛化。
- 两张训练图的 10 条指令可过拟合至 Top-1 100%，13 对指令同时成功率 100%。这只验证优化与条件输入，不代表测试性能。
- 历史真实 pilot 报告：`runs/real_pilot_001/REPORT.zh-CN.html`；测试预测框/热图：`runs/real_pilot_001/swap/test_visuals/index.html`。该 pilot 只用于初步诊断，不能代替提交版的三个种子。
- 主训练入口现支持由 `scripts/run_submission.py` 启动六个单 GPU run；提交版 batch size 为 256，并留出 GPU 6 做独立评测。

## 环境

服务器目录：`/root/labron/swapgrasp`。`.venv` 通过 `--system-site-packages` 复用已存在的 CUDA PyTorch，新增包只装在本项目环境；现有其他项目环境不作修改。当前验证环境：PyTorch 2.7.1+cu126 / torchvision 0.22.1+cu126，NumPy 固定 2.2.6。实际完整版本记录在 `artifacts/requirements-resolved.txt`。

```bash
cd /root/labron/swapgrasp
source .venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0
# 全新机器先安装适合 GPU 的 torch/torchvision，再执行：
pip install -e '.[test]'
```

## 数据与来源

授权来源：[Grasp-Anything](https://huggingface.co/datasets/airvlab/Grasp-Anything) 和 [Grasp-Anything++](https://huggingface.co/datasets/airvlab/Grasp-Anything-pp)。镜像只作为传输入口，仓库 revision 和 LFS SHA-256 均记录。下载正标注、指令、部件 mask、基础物体 mask、场景说明以及两段图像，总计 **77,855,053,299 字节**。本方法不依赖负抓取包和 6D 数据。

```bash
python download_data.py --workers 4
tail -f logs/download.log
```

文件先写 `.part`，成功校验后才重命名并生成 `.verified.json`。同一个目标文件不要同时启动两个下载进程。完整图像 ZIP 由 `image_part_aa`、`image_part_ab` 顺序拼接组成；我们的远程 ZIP reader 可以按 range 获取子集图像，避免为了先做实验再复制一个 65 GB 合并包。子集图像由 ZIP CRC 验证，单独记录为“全档 SHA 待完成”；不能声称这些图像已通过全档 SHA。

官方协议清单参考 `third_party/LGD/split/grasp-anything++`，上游 [andvg3/LGD](https://github.com/andvg3/LGD)，当前参考 commit `501904dc5c30c8fad58d90f7e4f49b4095f11eb0`。参考仓库只提供数据协议和格式参照，不参与本项目模型前向。

已发现官方 train 与 unseen-test 清单有 **37 张相同图像**。本项目 adapter 从训练侧移除测试图像，并以图像为单位划 validation，协议明确命名 `image_disjoint_diagnostic_subset`；它不是原样官方协议复现，也不是全数据实验。

```bash
# 正式子集要求四类标注包全档 SHA 已校验；小型诊断可显式允许逐成员 CRC 校验的下载前缀
python prepare_real.py --allow-prefix-annotations --output data/pilot_prefix --train-images 16 --val-images 4 --test-images 8
# 通过审计且检查联系表后，准备全档校验的诊断子集
python prepare_real.py --train-images 100 --val-images 20 --test-images 40
python tools.py audit --manifest data/processed/manifest.jsonl
python visualize.py --manifest data/processed/manifest.jsonl --split train --output artifacts/train_contact_sheet.html
python scripts/contact_sheet.py --manifest data/processed/manifest.jsonl --split train --output artifacts/train_contact_sheet.png
```

已核实的原始结构：指令为逐目标 `.pkl` 字符串，正抓取为 `[N,6]` Tensor（第一列为分数），部件/物体 mask 为独立二维二值 `.npy`。内部角度采用 x 向右、y 向下的顺时针弧度。由于官方绘图函数的自身符号约定，内部角度等于原始 degree 的弧度转换；相关顶点一致性有单元测试。

Manifest 每条记录保留准确指令、全部合法 GT、image/object/part ID、来源、split 和标注完整性。当前配对只做**不同物体、同粒度区域**，要求区域和抓取中心支持集合均互斥；同物体不同部件配对留到后续。没有配对不会伪造负例。

## 训练与公平对照

正式权重由 `weights/resnet18-f37072fd.pth` 和 `weights/clip-vit-b32-openai.bin` 提供；正式配置包含这些路径。训练前检查两个文件和对应的校验记录。图像权重来自 torchvision 官方地址，CLIP 权重来自 `timm/vit_base_patch32_clip_224.openai` 的 open_clip checkpoint。

```bash
python train.py --config configs/baseline.yaml --output runs/baseline_s42
python train.py --config configs/swap.yaml --output runs/swap_s42
# 使用完全相同的配置恢复；不能修改 max_steps 后冒充精确续训
python train.py --config configs/swap.yaml --output runs/swap_s42 --resume runs/swap_s42/last.pt
tensorboard --logdir runs --host 127.0.0.1 --port 6006
```

两个正式配置仅 `loss.lambda_swap` 不同，采样器、初始化、query 数和步数一致。每步 4 个普通 query 加 2 对 query，同图视觉特征只计算一次；训练不缓存可训练的视觉特征。选择 `best.pt` 只用 validation grasp@1，平分保留较早 checkpoint。日志包含每步样本 ID 序列，恢复包含 optimizer、scheduler、scaler 和所有 RNG 状态。

AMP 在支持 BF16 的 GPU（当前 A100）上默认使用 BF16，其他 CUDA GPU 使用 FP16；可通过 `train.amp_dtype` 显式指定，实际模式记录在 `environment.json`。BF16 保留较宽指数范围，避免稀疏中心监督下初始背景损失导致 FP16 梯度溢出；GradScaler 仅用于 FP16。

早期真实数据先用 overfit/pilot 检查；当前正式多种子协议与自动评测见文首和 `REPRODUCE.zh-CN.md`。Pilot 脚本及其独立结果目录仍保留作诊断记录，不覆盖提交版目录。

```bash
# 当前下载与训练状态，不加载 CUDA
python scripts/status.py
# 查看首轮真实数据实验
cat runs/real_pilot_001/REPORT.zh-CN.md
# 重新生成中文网页报告
python scripts/report_pilot.py --run runs/real_pilot_001
```

源数据存在文本/目标错配，以及不同部件指令共享标签的情况。当前 adapter 的 `mapping_verified` 只证明官方文件名映射；`semantic_verified` 不会据此自动设为 true。早期 manifest 中的旧字段仅表示映射，报告已注明。几何配对筛选不等于人工确认文本语义。

## 推理、评测、可视化

```bash
python evaluate.py predict --checkpoint runs/swap_s42/best.pt --manifest data/processed/manifest.jsonl --split test --output runs/swap_s42/test_predictions.jsonl --visualize-dir runs/swap_s42/prediction_visuals
python evaluate.py score --manifest data/processed/manifest.jsonl --split test --predictions runs/swap_s42/test_predictions.jsonl --pairs data/processed/train_pairs.jsonl --output runs/swap_s42/test
python visualize.py --manifest data/processed/manifest.jsonl --predictions runs/swap_s42/test_predictions.jsonl --output runs/swap_s42/gallery.html
python visualize.py --log runs/swap_s42/metrics.jsonl --output runs/swap_s42/curves.png
python explain.py --checkpoint runs/swap_s42/best.pt --baseline-checkpoint runs/baseline_s42/best.pt --manifest data/processed/manifest.jsonl --pairs data/processed/train_pairs.jsonl --predictions runs/swap_s42/test_predictions.jsonl --output runs/swap_s42/explanations
python evaluate.py compare --scores-a runs/baseline_s42/test/per_query.jsonl --scores-b runs/swap_s42/test/per_query.jsonl --output runs/comparison.json
```

`predict` 不读取 GT 或目标 mask。抓取成功要求**同一个 GT** 同时满足 IoU > 0.25 和角度误差 < 30°，报告 Top-1/5、query micro/image macro、同图两指令同时成功率，以及有对应标注时的 seen/unseen/H。对照区间以图像聚类进行 paired bootstrap。旋转 IoU 使用连续多边形交集，未与官方像素光栅化 evaluator 逐样本核对前，不能将二者数值视为完全相同。

`predict --visualize-dir` 同时输出逐指令的预测抓取框 PNG、质量热图 PNG 和 `index.html`；图上显示指令和 top-1 的位置、尺寸、角度及分数，坐标恢复到原始图像。这个入口仅需 RGB、文本和 ID；需要绿色 GT 对照图时使用单独的 `visualize.py`。

可视化包含可筛选预测页面、训练曲线 PNG/SVG、固定 `[0,1]` 色标的质量图、同图两指令 × 两模型的 2×2 比较、失败 CSV/PNG。绿色为全部 GT，红色为 top-1，黄色为其他候选。失败标签只是基于标注的诊断，不是物理执行成功判定。

## 工程检查

```bash
python scripts/verify.py
python scripts/resume_check.py
```

工程检查覆盖几何周期/变换、同一个 GT 双阈值、GT 多模式完整向量匹配、空配对及梯度方向、同图共享特征、推理不访问标签、划分泄漏与精确采样续训。合成样本指标不作为研究结果。本次真实数据已完成多 GPU 吞吐/显存基准，正式训练结果应由独立测试和真实数据报告给出；部件语言泛化与官方 evaluator 数值一致性仍有限制。

损失的质量背景使用闭世界标注近似：未标注的位置按当前任务背景训练，并不表示其物理上不可抓。明确未知区域可通过 `ignore_mask` 排除。
