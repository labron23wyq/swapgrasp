# 优化实验复现

代码副本在服务器 /root/labron/swapgrasp/optimization_001，使用父目录现有 .venv。
本轮数据与结果分别位于父目录 data/optimization_001 和 runs/optimization_001。
旧 submission_001 的代码、权重、测试和两页论文保留。

固定比较 original、balanced、hard_swap 三组，每组 seed 42、123，3500 步，batch 256。
original 使用 50% 配对 query；balanced/hard_swap 使用 12.5% 配对 query 并按图像均衡。
hard_swap 在第 500 步后线性加入，1000 步达到 lambda=0.05；对竞争抓取区域的最高四分之一像素排序，已满足 margin=0.5 的方向不再优化。
balanced 和 hard_swap 共享初始化和样本序列。original 对比 balanced 同时改变配对比例及图像均衡，不能将其收益独立归因于任意一个因素。

图像划分为 train 6580、val 353、rankfit 351、test 897。
该 test 来自旧 train，但本轮从 ImageNet/CLIP 预训练重新训练，未用旧 checkpoint 初始化。
它适合本轮诊断对照；旧实验曾使用过这些图像，研究方向也受旧结果启发，因此不是完全独立确认性 benchmark。
val 选择检测器 checkpoint；rankfit 只训练冻结检测器的候选重排头；val 选择重排 epoch 和残差权重。
重排头比较 0、0.25、0.5、1、2 权重，0 为保留原排序的回退项。设置在读取本轮 test 前固定。

重排头读取候选旋转区域的 3×3 特征、中心特征、文本特征、位置/尺寸/角度与原始热图分数。
训练预测标准与严格抓取成功，并加入同一指令候选之间的列表排序损失。它只重排原五个框，Top-5 必须不变。
GT 仅用来构造训练标签和离线评分，不进入候选生成或特征抽取。

在代码目录运行 ../.venv/bin/python infer_reranked.py，参数为：

    --detector ../runs/optimization_001/balanced_s42/best.pt
    --ranker ../runs/optimization_001/reranked_s42/best.pt
    --image /absolute/path/image.jpg
    --text "grasp the blue bottle"
    --output ../artifacts/reranked_demo

独立推理输入只有图片和文本，输出原图坐标框及前后排序可视化。
原始 score 仍是 detector 的中心分数，候选顺序由重排残差决定；它不是机器人抓取成功概率。
训练 loss、验证成功率、显存和吞吐写入 TensorBoard；
python scripts/optimization_workflow.py 会生成静态训练曲线、逐条评分、bootstrap、可视化对照和报告。
geometry 局部精修尚未包含在本轮固定实验中，不能声称已验证该方向。

重排头使用 detector baseline 未使用的 rankfit 标签；若有提升，它是额外监督加重排的组合效果。
不同采样产生的训练 loss、不同公式的 swap loss，不宜直接比较绝对大小。
